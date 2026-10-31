// Metal ownership and command submission. CPU hashing stays in Rust.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "shader.inc"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <unistd.h>

namespace {
static void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
// The device, its one command queue and the pipelines, shared by every
// hasher of the process: callers on several threads then feed one queue,
// and the shader compiles once.
struct Engine {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> leaf, parent;
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
    }
    static Engine &shared() {
        static Engine engine;
        return engine;
    }
};
struct Metal {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> leaf, parent;
    id<MTLBuffer> input, a, b;
    size_t capacity = 0;
    // A stream's slots: a message's own buffers, kept and reused, and its
    // command while one is in flight.
    struct Slot {
        id<MTLBuffer> input, a, b, result;
        id<MTLCommandBuffer> cmd;
        size_t capacity = 0;
    };
    std::vector<Slot> slots;

    Metal() {
        Engine &engine = Engine::shared();
        device = engine.device;
        queue = engine.queue;
        leaf = engine.leaf;
        parent = engine.parent;
    }
    void reserve(size_t length) {
        size_t bytes = std::max(size_t(1), length);
        require(bytes <= device.maxBufferLength, "Input exceeds Metal maxBufferLength");
        require((bytes - 1) / 1024 + 1 <= UINT32_MAX, "Input exceeds Metal dispatch limit");
        if (bytes <= capacity) return;
        size_t cvbytes = ((bytes - 1) / 1024 + 1) * 32;
        a = [device newBufferWithLength:cvbytes options:MTLResourceStorageModeShared];
        b = [device newBufferWithLength:cvbytes options:MTLResourceStorageModeShared];
        require(a && b, "Metal buffer allocation failed");
        capacity = bytes;
    }
    id<MTLCommandBuffer> command() {
        auto cmd = [queue commandBuffer];
        require(cmd != nil, "Metal command buffer allocation failed");
        return cmd;
    }
    void encode(id<MTLCommandBuffer> cmd, id<MTLComputePipelineState> pipeline,
                id<MTLBuffer> in, id<MTLBuffer> out, uint64_t length, uint64_t first, size_t threads, uint64_t base = 0) {
        auto encoder = [cmd computeCommandEncoder];
        require(encoder != nil, "Metal encoder allocation failed");
        uint64_t args[] = {length, first, base};
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:in offset:0 atIndex:0];
        [encoder setBuffer:out offset:0 atIndex:1];
        [encoder setBytes:args length:sizeof(args) atIndex:2];
        auto width = std::min(NSUInteger(128), pipeline.maxTotalThreadsPerThreadgroup);
        [encoder dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [encoder endEncoding];
    }
    // Reduce count nodes by a factor of width (a power of two) from node
    // first * width on, a level a pass: count and first become the nodes'
    // count and first index after. Four levels a pass in one thread ran
    // slower (fifteen serial compressions on few threads).
    void reduce(id<MTLCommandBuffer> cmd, id<MTLBuffer> __strong &src, id<MTLBuffer> __strong &dst, size_t &count, size_t &first, size_t width) {
        for (; width > 1; width /= 2) {
            first /= 2;
            encode(cmd, parent, src, dst, count, first, (count + 1) / 2 - first);
            count = (count + 1) / 2;
            std::swap(src, dst);
        }
    }
    void wait(id<MTLCommandBuffer> cmd) {
        [cmd waitUntilCompleted];
        require(cmd.status == MTLCommandBufferStatusCompleted,
                cmd.error ? cmd.error.localizedDescription.UTF8String : "Metal execution failed");
    }
    // callback is synchronous, cannot unwind, and writes cpu_groups CVs to out.
    // data and context remain valid throughout this call; no pointer is retained.
    void hash(const void *data, size_t length, size_t cpu_groups, size_t group_chunks,
              void (*cpu)(void *, uint8_t *), void *context, uint8_t *digest) {
        reserve(length);
        size_t first = cpu_groups * group_chunks;
        size_t offset = first * 1024;
        size_t count = length ? (length - 1) / 1024 + 1 : 1;
        auto cmd = command();
        // The GPU's share is copied into the hasher's own buffer: mapping
        // the caller's pages instead deferred about a millisecond of driver
        // work a call, outside the process, and slowed the caller's next
        // writes to those pages.
        size_t bytes = std::max(size_t(1), length - offset);
        if (!input || input.length < bytes) {
            input = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            require(input != nil, "Metal input allocation failed");
        }
        if (length > offset) memcpy(input.contents, static_cast<const uint8_t *>(data) + offset, length - offset);
        encode(cmd, leaf, input, a, length, offset / 1024, count - offset / 1024, offset);
        id<MTLBuffer> src = a, dst = b;
        // Reduce the GPU suffix to subtrees of the same size as the CPU
        // pool produces. Prefix slots remain unused until both devices finish.
        if (cpu_groups) {
            reduce(cmd, src, dst, count, first, group_chunks);
            std::vector<uint8_t> cpu_cvs(cpu_groups * 32);
            [cmd commit];
            // Metal requires synchronization even for writes to disjoint regions.
            cpu(context, cpu_cvs.data());
            wait(cmd);
            memcpy(src.contents, cpu_cvs.data(), cpu_cvs.size());
            cmd = command();
        }
        size_t zero = 0;
        while (count > 1) reduce(cmd, src, dst, count, zero, 2);
        [cmd commit];
        wait(cmd);
        memcpy(digest, src.contents, 32);
    }
    // Probe: hash length bytes at data, whole on the GPU, read in place
    // through a buffer wrapping the caller's pages (no copy). data is page
    // aligned, and the pages up to the next page boundary are the caller's.
    void hash_nocopy(const void *data, size_t length, uint8_t *digest) {
        size_t page = size_t(getpagesize());
        require(length > 0 && reinterpret_cast<uintptr_t>(data) % page == 0, "no-copy input must be page aligned and nonempty");
        reserve(length);
        size_t wrapped = (length + page - 1) / page * page;
        id<MTLBuffer> in = [device newBufferWithBytesNoCopy:const_cast<void *>(data) length:wrapped options:MTLResourceStorageModeShared deallocator:nil];
        require(in != nil, "Metal no-copy wrap failed");
        size_t count = (length - 1) / 1024 + 1;
        auto cmd = command();
        encode(cmd, leaf, in, a, length, 0, count);
        id<MTLBuffer> src = a, dst = b;
        size_t zero = 0;
        while (count > 1) reduce(cmd, src, dst, count, zero, 2);
        [cmd commit];
        wait(cmd);
        memcpy(digest, src.contents, 32);
    }
    // Probe: wrap the caller's pages in a buffer and release it, no command.
    void wrap_only(const void *data, size_t length) {
        size_t page = size_t(getpagesize());
        require(length > 0 && reinterpret_cast<uintptr_t>(data) % page == 0, "no-copy input must be page aligned and nonempty");
        size_t wrapped = (length + page - 1) / page * page;
        id<MTLBuffer> in = [device newBufferWithBytesNoCopy:const_cast<void *>(data) length:wrapped options:MTLResourceStorageModeShared deallocator:nil];
        require(in != nil, "Metal no-copy wrap failed");
    }
    // The slot's input memory, holding capacity bytes: allocated when the
    // slot is new or too small. The slot has no command in flight.
    uint8_t *stream_reserve(size_t slot, size_t capacity) {
        if (slots.size() <= slot) slots.resize(slot + 1);
        Slot &s = slots[slot];
        require(s.cmd == nil, "Metal stream slot is in flight");
        size_t bytes = std::max(size_t(1), capacity);
        require(bytes <= device.maxBufferLength, "Input exceeds Metal maxBufferLength");
        require((bytes - 1) / 1024 + 1 <= UINT32_MAX, "Input exceeds Metal dispatch limit");
        if (s.capacity < bytes) {
            size_t cvbytes = ((bytes - 1) / 1024 + 1) * 32;
            s.input = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            s.a = [device newBufferWithLength:cvbytes options:MTLResourceStorageModeShared];
            s.b = [device newBufferWithLength:cvbytes options:MTLResourceStorageModeShared];
            require(s.input && s.a && s.b, "Metal buffer allocation failed");
            s.capacity = bytes;
        }
        return static_cast<uint8_t *>(s.input.contents);
    }
    // Hash the slot's first length bytes, whole on the GPU, without waiting.
    void stream_submit(size_t slot, size_t length) {
        Slot &s = slots.at(slot);
        require(s.cmd == nil && length <= s.capacity, "Metal stream slot is in flight or too small");
        size_t count = length ? (length - 1) / 1024 + 1 : 1;
        auto cmd = command();
        encode(cmd, leaf, s.input, s.a, length, 0, count);
        id<MTLBuffer> src = s.a, dst = s.b;
        size_t zero = 0;
        while (count > 1) reduce(cmd, src, dst, count, zero, 2);
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
extern "C" void *blake3_servil_metal_create(char *error) {
    @autoreleasepool { try { return new Metal; }
    catch (const std::exception &e) { snprintf(error, 512, "%s", e.what()); return nullptr; } }
}
extern "C" void blake3_servil_metal_destroy(void *ctx) { delete static_cast<Metal *>(ctx); }
extern "C" uint8_t *blake3_servil_metal_stream_reserve(void *ctx, size_t slot, size_t capacity, char *error) {
    @autoreleasepool { try { return static_cast<Metal *>(ctx)->stream_reserve(slot, capacity); }
    catch (const std::exception &e) { snprintf(error, 512, "%s", e.what()); return nullptr; } }
}
extern "C" bool blake3_servil_metal_stream_submit(void *ctx, size_t slot, size_t length, char *error) {
    @autoreleasepool { try { static_cast<Metal *>(ctx)->stream_submit(slot, length); return true; }
    catch (const std::exception &e) { snprintf(error, 512, "%s", e.what()); return false; } }
}
extern "C" bool blake3_servil_metal_stream_done(void *ctx, size_t slot) {
    @autoreleasepool { return static_cast<Metal *>(ctx)->stream_done(slot); }
}
extern "C" bool blake3_servil_metal_stream_wait(void *ctx, size_t slot, uint8_t *out, char *error) {
    @autoreleasepool { try { static_cast<Metal *>(ctx)->stream_wait(slot, out); return true; }
    catch (const std::exception &e) { snprintf(error, 512, "%s", e.what()); return false; } }
}
extern "C" bool blake3_servil_metal_hash(void *ctx, const void *data, size_t length, size_t cpu_groups, size_t group_chunks,
                                void (*cpu)(void *, uint8_t *), void *context, uint8_t *out, char *error) {
    @autoreleasepool { try {
        static_cast<Metal *>(ctx)->hash(data, length, cpu_groups, group_chunks, cpu, context, out);
        return true;
    } catch (const std::exception &e) { snprintf(error, 512, "%s", e.what()); return false; } }
}
extern "C" bool blake3_servil_metal_hash_nocopy(void *ctx, const void *data, size_t length, uint8_t *out, char *error) {
    @autoreleasepool { try { static_cast<Metal *>(ctx)->hash_nocopy(data, length, out); return true; }
    catch (const std::exception &e) { snprintf(error, 512, "%s", e.what()); return false; } }
}
extern "C" bool blake3_servil_metal_wrap_only(void *ctx, const void *data, size_t length, char *error) {
    @autoreleasepool { try { static_cast<Metal *>(ctx)->wrap_only(data, length); return true; }
    catch (const std::exception &e) { snprintf(error, 512, "%s", e.what()); return false; } }
}
