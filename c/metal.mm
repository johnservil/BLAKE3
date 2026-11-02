// Metal ownership and command submission; the CPU's share of a stream
// stays in Rust.
#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>
#import <Metal/Metal.h>
#include "shader.inc"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
static void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
// The GPU's core count, from the accelerator's registry entry.
static uint32_t gpu_cores() {
    io_service_t service = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AGXAccelerator"));
    require(service != IO_OBJECT_NULL, "no Apple GPU accelerator in the IO registry");
    CFTypeRef value = IORegistryEntryCreateCFProperty(service, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
    IOObjectRelease(service);
    int64_t cores = 0;
    bool ok = value && CFGetTypeID(value) == CFNumberGetTypeID() && CFNumberGetValue((CFNumberRef)value, kCFNumberSInt64Type, &cores);
    if (value) CFRelease(value);
    require(ok && cores > 0, "the GPU's core count is not in the IO registry");
    return uint32_t(cores);
}
// The device, its one command queue and the pipelines, shared by every
// hasher of the process: callers on several threads then feed one queue,
// and the shader compiles once.
struct Engine {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> leaf, parent;
    uint32_t cores;
    Engine() {
        device = MTLCreateSystemDefaultDevice();
        require(device && device.hasUnifiedMemory, "Metal requires an Apple Silicon GPU");
        queue = [device newCommandQueue];
        require(queue != nil, "Metal command queue allocation failed");
        NSError *error = nil;
        auto library = [device newLibraryWithSource:@(shader_source) options:nil error:&error];
        require(library != nil, error ? error.localizedDescription.UTF8String : "Metal compilation failed");
        leaf = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"leaves"] error:&error];
        require(leaf != nil, "Metal leaf pipeline failed");
        parent = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"parents"] error:&error];
        require(parent != nil, "Metal parent pipeline failed");
        cores = gpu_cores();
    }
    static Engine &shared() {
        static Engine engine;
        return engine;
    }
};
struct Args { uint64_t length, first; uint32_t key[8], mode; };
// A stream's slots: a message's own buffers, kept and reused, and its
// command while one is in flight.
struct Slot {
    id<MTLBuffer> input, a, b, result;
    id<MTLCommandBuffer> cmd;
    size_t capacity = 0;
};
struct Metal {
    Engine &engine;
    std::vector<Slot> slots;

    Metal() : engine(Engine::shared()) {}
    id<MTLCommandBuffer> command() {
        auto cmd = [engine.queue commandBuffer];
        require(cmd != nil, "Metal command buffer allocation failed");
        return cmd;
    }
    void encode(id<MTLCommandBuffer> cmd, id<MTLComputePipelineState> pipeline, id<MTLBuffer> in, id<MTLBuffer> out, Args args, size_t threads) {
        auto encoder = [cmd computeCommandEncoder];
        require(encoder != nil, "Metal encoder allocation failed");
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:in offset:0 atIndex:0];
        [encoder setBuffer:out offset:0 atIndex:1];
        [encoder setBytes:&args length:sizeof(args) atIndex:2];
        auto width = std::min(NSUInteger(128), pipeline.maxTotalThreadsPerThreadgroup);
        [encoder dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [encoder endEncoding];
    }
    void wait(id<MTLCommandBuffer> cmd) {
        [cmd waitUntilCompleted];
        require(cmd.status == MTLCommandBufferStatusCompleted,
                cmd.error ? cmd.error.localizedDescription.UTF8String : "Metal execution failed");
    }
    // The slot's input memory, holding capacity bytes: allocated when the
    // slot is new or too small. The slot has no command in flight.
    uint8_t *stream_reserve(size_t slot, size_t capacity) {
        if (slots.size() <= slot) slots.resize(slot + 1);
        Slot &s = slots[slot];
        require(s.cmd == nil, "Metal stream slot is in flight");
        size_t bytes = std::max(size_t(1), capacity);
        require(bytes <= engine.device.maxBufferLength, "Input exceeds Metal maxBufferLength");
        require((bytes - 1) / 1024 + 1 <= UINT32_MAX, "Input exceeds Metal dispatch limit");
        if (s.capacity < bytes) {
            size_t cvbytes = ((bytes - 1) / 1024 + 1) * 32;
            s.input = [engine.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            s.a = [engine.device newBufferWithLength:cvbytes options:MTLResourceStorageModeShared];
            s.b = [engine.device newBufferWithLength:cvbytes options:MTLResourceStorageModeShared];
            require(s.input && s.a && s.b, "Metal buffer allocation failed");
            s.capacity = bytes;
        }
        return static_cast<uint8_t *>(s.input.contents);
    }
    // Hash the slot's first length bytes, whole on the GPU in the mode
    // (key and flags), without waiting: the leaves, then a tree level a
    // pass, an odd last node carried up (four levels a pass in one thread
    // ran slower: fifteen serial compressions on few threads).
    void stream_submit(size_t slot, size_t length, const uint32_t *key, uint32_t mode) {
        Slot &s = slots.at(slot);
        require(s.cmd == nil && length <= s.capacity, "Metal stream slot is in flight or too small");
        size_t count = length ? (length - 1) / 1024 + 1 : 1;
        Args args = {length, 0, {}, mode};
        memcpy(args.key, key, sizeof(args.key));
        auto cmd = command();
        encode(cmd, engine.leaf, s.input, s.a, args, count);
        id<MTLBuffer> src = s.a, dst = s.b;
        while (count > 1) {
            args.length = count;
            encode(cmd, engine.parent, src, dst, args, (count + 1) / 2);
            count = (count + 1) / 2;
            std::swap(src, dst);
        }
        s.result = src;
        [cmd commit];
        s.cmd = cmd;
    }
    bool stream_done(size_t slot) {
        Slot &s = slots.at(slot);
        require(s.cmd != nil, "Metal stream slot has no command in flight");
        return s.cmd.status >= MTLCommandBufferStatusCompleted;
    }
    // Wait for the slot's command and take its digest; the slot is free after.
    void stream_wait(size_t slot, uint8_t *digest) {
        Slot &s = slots.at(slot);
        require(s.cmd != nil, "Metal stream slot has no command in flight");
        wait(s.cmd);
        s.cmd = nil;
        memcpy(digest, s.result.contents, 32);
    }
};

} // namespace

// No C++ exception crosses the Rust FFI. error points to 512 writable bytes.
extern "C" void *blake3_servil_metal_create(uint32_t *cores, char *error) {
    @autoreleasepool { try { auto metal = new Metal; *cores = metal->engine.cores; return metal; }
    catch (const std::exception &e) { snprintf(error, 512, "%s", e.what()); return nullptr; } }
}
extern "C" void blake3_servil_metal_destroy(void *ctx) { delete static_cast<Metal *>(ctx); }
extern "C" uint8_t *blake3_servil_metal_stream_reserve(void *ctx, size_t slot, size_t capacity, char *error) {
    @autoreleasepool { try { return static_cast<Metal *>(ctx)->stream_reserve(slot, capacity); }
    catch (const std::exception &e) { snprintf(error, 512, "%s", e.what()); return nullptr; } }
}
extern "C" bool blake3_servil_metal_stream_submit(void *ctx, size_t slot, size_t length, const uint32_t *key, uint32_t mode, char *error) {
    @autoreleasepool { try { static_cast<Metal *>(ctx)->stream_submit(slot, length, key, mode); return true; }
    catch (const std::exception &e) { snprintf(error, 512, "%s", e.what()); return false; } }
}
extern "C" bool blake3_servil_metal_stream_done(void *ctx, size_t slot) {
    @autoreleasepool { return static_cast<Metal *>(ctx)->stream_done(slot); }
}
extern "C" bool blake3_servil_metal_stream_wait(void *ctx, size_t slot, uint8_t *out, char *error) {
    @autoreleasepool { try { static_cast<Metal *>(ctx)->stream_wait(slot, out); return true; }
    catch (const std::exception &e) { snprintf(error, 512, "%s", e.what()); return false; } }
}
