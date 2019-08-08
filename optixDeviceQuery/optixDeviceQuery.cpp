/* 
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 *  optixDeviceQuery.cpp -- Demonstration of the device query functions
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include <optix.h>
#include <sutil.h>

int main(int argc, char *argv[])
{
    RTcontext context = 0;

    try
    {
        unsigned int num_devices;
        unsigned int version;
        unsigned int i;
        unsigned int context_device_count;
        unsigned int max_textures;
        int* context_devices;

        RT_CHECK_ERROR(rtDeviceGetDeviceCount(&num_devices));
        RT_CHECK_ERROR(rtGetVersion(&version));

        printf("OptiX %d.%d.%d\n", version / 10000, (version % 10000) / 100, version % 100); // major.minor.micro

        printf("Number of Devices = %d\n\n", num_devices);

        for(i = 0; i < num_devices; ++i) {
            char name[256];
            char pciBusId[16];
            int computeCaps[2];
            RTsize total_mem;
            int clock_rate;
            int threads_per_block;
            int sm_count;
            int execution_timeout_enabled;
            int texture_count;
            int tcc_driver;
            int cuda_device_ordinal;

            RT_CHECK_ERROR(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_NAME, sizeof(name), name));
            RT_CHECK_ERROR(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_PCI_BUS_ID, sizeof(pciBusId), pciBusId));
            printf("Device %d (%s): %s\n", i, pciBusId, name);

            RT_CHECK_ERROR(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY, sizeof(computeCaps), &computeCaps));
            printf("  Compute Support: %d %d\n", computeCaps[0], computeCaps[1]);

            RT_CHECK_ERROR(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_TOTAL_MEMORY, sizeof(total_mem), &total_mem));
            printf("  Total Memory: %llu bytes\n", (unsigned long long)total_mem);

            RT_CHECK_ERROR(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_CLOCK_RATE, sizeof(clock_rate), &clock_rate));
            printf("  Clock Rate: %u kilohertz\n", clock_rate);

            RT_CHECK_ERROR(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, sizeof(threads_per_block), &threads_per_block));
            printf("  Max. Threads per Block: %u\n", threads_per_block);

            RT_CHECK_ERROR(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, sizeof(sm_count), &sm_count));
            printf("  SM Count: %u\n", sm_count);

            RT_CHECK_ERROR(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_EXECUTION_TIMEOUT_ENABLED, sizeof(execution_timeout_enabled), &execution_timeout_enabled));
            printf("  Execution Timeout Enabled: %d\n", execution_timeout_enabled);

            RT_CHECK_ERROR(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_MAX_HARDWARE_TEXTURE_COUNT, sizeof(texture_count), &texture_count));
            printf("  Max. HW Texture Count: %u\n", texture_count);

            RT_CHECK_ERROR(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_TCC_DRIVER, sizeof(tcc_driver), &tcc_driver));
            printf("  TCC driver enabled: %u\n", tcc_driver);

            RT_CHECK_ERROR(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL, sizeof(cuda_device_ordinal), &cuda_device_ordinal));
            printf("  CUDA Device Ordinal: %d\n", cuda_device_ordinal);

            std::vector<int> compatible_devices;
			compatible_devices.resize(num_devices + 1);
            RT_CHECK_ERROR(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_COMPATIBLE_DEVICES, sizeof(compatible_devices), &compatible_devices[0]));
            printf("  Compatible devices: ");
            if (compatible_devices[0] == 0) {
                printf("none\n");
            } else {
                for(int i = 0; i < compatible_devices[0]; ++i) {
                    if( i > 0 ) {
                        printf(", ");
                    }
                    printf("%d", compatible_devices[i+1]);
                }
                printf("\n");
            }

            printf("\n");
        }

        printf("Constructing a context...\n");
        RT_CHECK_ERROR(rtContextCreate(&context));

        RT_CHECK_ERROR(rtContextGetDeviceCount(context, &context_device_count));
        printf("  Created with %u device(s)\n", context_device_count);

        RT_CHECK_ERROR(rtContextGetAttribute(context, RT_CONTEXT_ATTRIBUTE_MAX_TEXTURE_COUNT, sizeof(max_textures), &max_textures));
        printf("  Supports %u simultaneous textures\n", max_textures);

        context_devices = (int*)malloc(sizeof(int)*context_device_count);
        RT_CHECK_ERROR(rtContextGetDevices(context, context_devices));

        printf("  Free memory:\n");
        for(i = 0; i < context_device_count; ++i) {
            int ordinal = context_devices[i];
            RTsize bytes;
            RT_CHECK_ERROR(rtContextGetAttribute(context, (RTcontextattribute)(RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY+ordinal), sizeof(bytes), &bytes));
            printf("    Device %d: %llu bytes\n", ordinal, (unsigned long long)bytes);
        }
        free(context_devices);

        printf("\n");

        return 0;

    } SUTIL_CATCH( context )
}
    
