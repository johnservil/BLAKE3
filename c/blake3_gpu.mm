// The GPU through Metal (src/gpu.rs): the device, its queue and the two
// pipelines, made once per process; and a stream storage's slots, each a
// message's bytes in memory the CPU and the GPU share (unified memory),
// with the buffers for its chaining values and its command while one is
// in flight. Hashing stays in the shader (c/blake3_gpu.metal); this file
// owns Metal objects and commits commands. No C++ exception crosses into
// Rust: each entry point catches and reports through `error` (512 bytes).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "blake3_gpu_shader.inc"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

struct Engine {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> leaves, parents;
};
// Set once by blake3_servil_gpu_start, before src/gpu.rs publishes the
// GPU as ready (its Release store, the readers' Acquire load).
Engine *engine = nullptr;

struct Args {
    uint64_t length;
    uint64_t first;
    uint32_t key[8];
    uint32_t flags;
};

struct Slot {
    id<MTLBuffer> input, a, b, result;
    id<MTLCommandBuffer> cmd;
    size_t capacity = 0;
};
struct Slots {
    std::vector<Slot> slots;
};

void encode(id<MTLCommandBuffer> cmd, id<MTLComputePipelineState> pipeline, id<MTLBuffer> in, id<MTLBuffer> out, const Args &args,
            size_t threads) {
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    require(encoder != nil, "Metal encoder allocation failed");
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:in offset:0 atIndex:0];
    [encoder setBuffer:out offset:0 atIndex:1];
    [encoder setBytes:&args length:sizeof(args) atIndex:2];
    NSUInteger width = std::min(NSUInteger(128), pipeline.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder endEncoding];
}
} // namespace

extern "C" bool blake3_servil_gpu_start(char *error) {
    @autoreleasepool {
        try {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            require(device != nil && device.hasUnifiedMemory, "no Metal device with unified memory");
            id<MTLCommandQueue> queue = [device newCommandQueue];
            require(queue != nil, "Metal command queue allocation failed");
            NSError *e = nil;
            id<MTLLibrary> library = [device newLibraryWithSource:@(blake3_gpu_shader) options:nil error:&e];
            require(library != nil, e ? e.localizedDescription.UTF8String : "Metal shader compilation failed");
            id<MTLComputePipelineState> leaves = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"leaves"] error:&e];
            require(leaves != nil, "Metal leaf pipeline failed");
            id<MTLComputePipelineState> parents = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"parents"] error:&e];
            require(parents != nil, "Metal parent pipeline failed");
            engine = new Engine{device, queue, leaves, parents};
            return true;
        } catch (const std::exception &x) {
            snprintf(error, 512, "%s", x.what());
            return false;
        }
    }
}

extern "C" void *blake3_servil_gpu_slots_new(size_t count) {
    Slots *s = new Slots;
    s->slots.resize(count);
    return s;
}

// The slot's bytes, at least `capacity` of them, allocated when the slot
// is new or smaller: the slot has no command in flight. Null on failure.
extern "C" uint8_t *blake3_servil_gpu_room(void *slots, size_t slot, size_t capacity, char *error) {
    @autoreleasepool {
        try {
            Slot &s = static_cast<Slots *>(slots)->slots.at(slot);
            require(s.cmd == nil, "a GPU slot with a command in flight");
            require(capacity >= 1 && capacity <= engine->device.maxBufferLength, "a message longer than the GPU's largest buffer");
            if (s.capacity < capacity) {
                size_t cvs = ((capacity - 1) / 1024 + 1) * 32;
                s.input = [engine->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
                s.a = [engine->device newBufferWithLength:cvs options:MTLResourceStorageModeShared];
                s.b = [engine->device newBufferWithLength:cvs options:MTLResourceStorageModeShared];
                require(s.input && s.a && s.b, "Metal buffer allocation failed");
                s.capacity = capacity;
            }
            return static_cast<uint8_t *>(s.input.contents);
        } catch (const std::exception &x) {
            snprintf(error, 512, "%s", x.what());
            return nullptr;
        }
    }
}

// Commit the hash of the slot's first `length` bytes (1 or more) under
// `key` and `flags`, without waiting.
extern "C" bool blake3_servil_gpu_submit(void *slots, size_t slot, size_t length, const uint32_t *key, uint32_t flags, char *error) {
    @autoreleasepool {
        try {
            Slot &s = static_cast<Slots *>(slots)->slots.at(slot);
            require(s.cmd == nil && length >= 1 && length <= s.capacity, "a GPU slot in flight or too small");
            Args args{length, 0, {}, flags};
            std::memcpy(args.key, key, sizeof(args.key));
            uint64_t count = (length - 1) / 1024 + 1;
            id<MTLCommandBuffer> cmd = [engine->queue commandBuffer];
            require(cmd != nil, "Metal command buffer allocation failed");
            encode(cmd, engine->leaves, s.input, s.a, args, count);
            id<MTLBuffer> src = s.a, dst = s.b;
            while (count > 1) {
                args.length = count;
                encode(cmd, engine->parents, src, dst, args, (count + 1) / 2);
                count = (count + 1) / 2;
                std::swap(src, dst);
            }
            s.result = src;
            [cmd commit];
            s.cmd = cmd;
            return true;
        } catch (const std::exception &x) {
            snprintf(error, 512, "%s", x.what());
            return false;
        }
    }
}

// Whether the slot's command has ended (or the slot has none).
extern "C" bool blake3_servil_gpu_done(void *slots, size_t slot) {
    @autoreleasepool {
        Slot &s = static_cast<Slots *>(slots)->slots[slot];
        return s.cmd == nil || s.cmd.status >= MTLCommandBufferStatusCompleted;
    }
}

// Wait for the slot's command and write its digest (32 bytes) to `out`;
// the slot is free after. False, with the reason, when the command failed.
extern "C" bool blake3_servil_gpu_wait(void *slots, size_t slot, uint8_t *out, char *error) {
    @autoreleasepool {
        try {
            Slot &s = static_cast<Slots *>(slots)->slots.at(slot);
            require(s.cmd != nil, "a GPU slot with no command");
            [s.cmd waitUntilCompleted];
            bool ok = s.cmd.status == MTLCommandBufferStatusCompleted;
            const char *why = ok ? nullptr : (s.cmd.error ? s.cmd.error.localizedDescription.UTF8String : "the GPU command failed");
            if (!ok) snprintf(error, 512, "%s", why);
            s.cmd = nil;
            std::memcpy(out, s.result.contents, 32);
            return ok;
        } catch (const std::exception &x) {
            snprintf(error, 512, "%s", x.what());
            return false;
        }
    }
}
