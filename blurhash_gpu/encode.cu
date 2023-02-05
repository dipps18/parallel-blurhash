#include "encode.cuh"
#include "common.cuh"

#include <string.h>

static char *encode_int(int value, int length, char *destination);

static int encodeDC(float r, float g, float b);
static int encodeAC(float r, float g, float b, float maximumValue);

class Color{
public:
	float* r;
	float* g;
	float* b;
};

__global__ void multiplyBasisFunction(int xComponent, int yComponent, int width, int height, uint8_t *rgb, size_t bytesPerRow, Color* color);

const char *blurHashForPixels(int xComponents, int yComponents, int width, int height, uint8_t *rgb, size_t bytesPerRow) {
	static char buffer[2 + 4 + (9 * 9 - 1) * 2 + 1];
	if(xComponents < 1 || xComponents > 9) return NULL;
	if(yComponents < 1 || yComponents > 9) return NULL;

	float factors[yComponents][xComponents][3];
	memset(factors, 0, sizeof(factors));
	Color *c;
	uint8_t *rgb_d;
	gpuErrchk(cudaMalloc(&rgb_d, sizeof(uint8_t)*width*height*3));
	gpuErrchk(cudaMemcpy(rgb_d, rgb, sizeof(uint8_t)*width*height*3, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMallocManaged(&c, sizeof(Color)));
	gpuErrchk(cudaMallocManaged(&c->r, sizeof(float)*width*height));
	gpuErrchk(cudaMallocManaged(&c->g, sizeof(float)*width*height));
	gpuErrchk(cudaMallocManaged(&c->b, sizeof(float)*width*height));

	for(int y = 0; y < yComponents; y++) {
		for(int x = 0; x < xComponents; x++) {
			int numThreads = 256;
			int numBlocks = (width * height + numThreads - 1)/numThreads;
			multiplyBasisFunction<<<numBlocks, numThreads>>>(x, y, width, height, rgb_d, bytesPerRow, c);
			gpuErrchk( cudaPeekAtLastError() );
			gpuErrchk(cudaDeviceSynchronize());
			factors[y][x][0] = thrust::reduce(c->r, c->r + width*height);
			factors[y][x][1] = thrust::reduce(c->g, c->g + width*height);
			factors[y][x][2] = thrust::reduce(c->b, c->b + width*height);
		}
	}

	float *dc = factors[0][0];
	float *ac = dc + 3;
	int acCount = xComponents * yComponents - 1;
	char *ptr = buffer;

	int sizeFlag = (xComponents - 1) + (yComponents - 1) * 9;
	ptr = encode_int(sizeFlag, 1, ptr);

	float maximumValue;
	if(acCount > 0) {
		float actualMaximumValue = 0;
		for(int i = 0; i < acCount * 3; i++) {
			actualMaximumValue = fmaxf(fabsf(ac[i]), actualMaximumValue);
		}

		int quantisedMaximumValue = fmaxf(0, fminf(82, floorf(actualMaximumValue * 166 - 0.5)));
		maximumValue = ((float)quantisedMaximumValue + 1) / 166;
		ptr = encode_int(quantisedMaximumValue, 1, ptr);
	} else {
		maximumValue = 1;
		ptr = encode_int(0, 1, ptr);
	}

	ptr = encode_int(encodeDC(dc[0], dc[1], dc[2]), 4, ptr);

	for(int i = 0; i < acCount; i++) {
		ptr = encode_int(encodeAC(ac[i * 3 + 0], ac[i * 3 + 1], ac[i * 3 + 2], maximumValue), 2, ptr);
	}

	*ptr = 0;
	gpuErrchk(cudaFree(c->r));
	gpuErrchk(cudaFree(c->b));
	gpuErrchk(cudaFree(c->g));
	gpuErrchk(cudaFree(c));
	gpuErrchk(cudaFree(rgb_d));
	return buffer;
}

__global__ 
void multiplyBasisFunction(int xComponent, int yComponent, int width, int height, uint8_t *rgb, size_t bytesPerRow, Color* c)  {
	
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if(tid < width*height)
	{
		float r = 0, g = 0, b = 0;
		float normalisation = (xComponent == 0 && yComponent == 0) ? 1 : 2;
		int x = tid % width;
		int y = tid / width;
		float basis = cosf(M_PI * xComponent * x / width) * cosf(M_PI * yComponent * y / height);

		r += basis * sRGBToLinear(rgb[3 * x + 0 + y * bytesPerRow]);
		g += basis * sRGBToLinear(rgb[3 * x + 1 + y * bytesPerRow]);
		b += basis * sRGBToLinear(rgb[3 * x + 2 + y * bytesPerRow]);

		float scale = normalisation / (width * height);
		c->r[tid] = r * scale;
		c->g[tid] = g * scale;
		c->b[tid] = b * scale;

	}
}


static int encodeDC(float r, float g, float b) {
	int roundedR = linearTosRGB(r);
	int roundedG = linearTosRGB(g);
	int roundedB = linearTosRGB(b);
	return (roundedR << 16) + (roundedG << 8) + roundedB;
}

static int encodeAC(float r, float g, float b, float maximumValue) {
	int quantR = fmaxf(0, fminf(18, floorf(signPow(r / maximumValue, 0.5) * 9 + 9.5)));
	int quantG = fmaxf(0, fminf(18, floorf(signPow(g / maximumValue, 0.5) * 9 + 9.5)));
	int quantB = fmaxf(0, fminf(18, floorf(signPow(b / maximumValue, 0.5) * 9 + 9.5)));

	return quantR * 19 * 19 + quantG * 19 + quantB;
}

static char characters[84] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz#$%*+,-.:;=?@[]^_{|}~";

static char *encode_int(int value, int length, char *destination) {
	int divisor = 1;
	for(int i = 0; i < length - 1; i++) divisor *= 83;

	for(int i = 0; i < length; i++) {
		int digit = (value / divisor) % 83;
		divisor /= 83;
		*destination++ = characters[digit];
	}
	return destination;
}