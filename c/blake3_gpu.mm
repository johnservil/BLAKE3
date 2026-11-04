// The GPU through Metal (src/gpu.rs): the device, its queue and the tasks
// pipeline, made once per process; buffers the CPU and the GPU share
// (unified memory); and the commit of a batch of a stream's tasks, whose
// completion calls back into src/gpu.rs. Hashing stays in the shader
// (c/blake3_gpu.metal); this file owns Metal objects and commits commands.
// No C++ exception crosses into Rust: each entry point catches and reports
// through `error` (512 bytes).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "blake3_gpu_shader.inc"
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace {
void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

struct Engine {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> tasks;
};
// Set once by blake3_servil_gpu_start, before src/gpu.rs publishes the
// GPU as ready (its Release store, the readers' Acquire load).
Engine *engine = nullptr;

struct Args {
    uint32_t key[8];
    uint32_t flags;
};
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
            id<MTLComputePipelineState> tasks = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"tasks"] error:&e];
            require(tasks != nil && tasks.maxTotalThreadsPerThreadgroup >= 64, "Metal tasks pipeline failed");
            engine = new Engine{device, queue, tasks};
            return true;
        } catch (const std::exception &x) {
            snprintf(error, 512, "%s", x.what());
            return false;
        }
    }
}

// A buffer of `length` bytes the CPU and the GPU share, kept for good (the
// caller never frees it): its handle, and its bytes at `*contents`. Null
// on failure.
extern "C" void *blake3_servil_gpu_buffer(size_t length, uint8_t **contents, char *error) {
    @autoreleasepool {
        id<MTLBuffer> buffer = [engine->device newBufferWithLength:length options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            snprintf(error, 512, "Metal buffer allocation failed (%zu bytes)", length);
            return nullptr;
        }
        *contents = static_cast<uint8_t *>(buffer.contents);
        return (__bridge_retained void *)buffer;
    }
}

// Commit the hash of `count` tasks (each a 64 KiB piece: its offset in
// `ring` and its chunk counter, 16 bytes a task in `pieces`, at most 4096
// bytes of them) under `key` and `flags`, 64 bytes of output a task into
// `out`, without waiting. When the command ends, Metal's thread calls
// `done(context, ok)`, with `ok` false where the command failed.
extern "C" bool blake3_servil_gpu_commit(void *ring, void *out, const void *pieces, size_t count, const uint32_t *key, uint32_t flags,
                                         void (*done)(void *, bool), void *context, char *error) {
    @autoreleasepool {
        try {
            require(count >= 1 && count * 16 <= 4096, "a batch of 1 to 256 tasks");
            Args args{{}, flags};
            std::memcpy(args.key, key, sizeof(args.key));
            id<MTLCommandBuffer> cmd = [engine->queue commandBufferWithUnretainedReferences];
            require(cmd != nil, "Metal command buffer allocation failed");
            id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
            require(encoder != nil, "Metal encoder allocation failed");
            [encoder setComputePipelineState:engine->tasks];
            [encoder setBuffer:(__bridge id<MTLBuffer>)ring offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
            [encoder setBytes:pieces length:count * 16 atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [encoder endEncoding];
            [cmd addCompletedHandler:^(id<MTLCommandBuffer> b) {
                done(context, b.status == MTLCommandBufferStatusCompleted);
            }];
            [cmd commit];
            return true;
        } catch (const std::exception &x) {
            snprintf(error, 512, "%s", x.what());
            return false;
        }
    }
}
