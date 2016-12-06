#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include "roughsort.h"
#include "wingetopt.h"
#include <iostream>

// for dealing with getopt arguments
extern char *optarg;
extern int optind;

void devCheckSortedness(int32_t* const a, const int n);

namespace {

	const int64_t MIN_ARRAY_LEN = 2,
		MAX_ARRAY_LEN = (1 << 30), // 4 GiB = ~1 billion 32-bit ints
		MIN_RAND_LEN = 1 << 10,
		MAX_RAND_LEN = 1 << 24;

	/* Parses an integer argument of the given radix from the command line, aborting
	after printing errMsg if an error occurs or the integer exceeds the given
	bounds. */
	int parseInt(int radix, int64_t min, int64_t max, const char* errMsg) {
		char* parsePtr = nullptr;
		auto l = strtoll(optarg, &parsePtr, radix);
		if ((size_t)(parsePtr - optarg) != strlen(optarg) || l < min || l > max ||
			ERANGE == errno) {
			fatal(errMsg);
		}
		return l;
	}


	[[noreturn]] void usage() {
		msg("roughsort [options]\n"
			"  -h        These instructions\n"
			"  -g        Skip the sequential, non-GPU algorithms\n"
			"  -k <int>  Set the k-sortedness of the random array (default: don't k-sort)\n"
			"  -m        Force usage of software Mersenne Twister RNG\n"
			"  -n <len>  Set randomized array length (default: random)\n"
			"  -s        Shuffle each (k + 1)-segment\n"
			"  -t        Confirm sorted array is in order\n");
		exit(1);
	}

	bool testArrayEq(const int32_t* const a, const int32_t* const b, const int n) {
		for (int i = 0; i < n; i++) {
			if (a[i] != b[i]) {
				return false;
			}
		}
		return true;
	}

	bool testArrayUniform(const int32_t* const a, const int n)
	{
		for (int i = 1; i < n; i++)
		{
			if (a[i] != a[0])
				return false;
		}
		return true;
	}

} // end anonymous namespace

int main(int argc, char* argv[]) {
	//msg("Roughsort Demonstration\nA. Pfaff, J. Treadwell 2016\n");

	bool runHostSorts = true;
	int k = -1;
	bool forceMT = false;
	int arrayLen = 0;
	bool shuffle = false;
	bool testSorted = false;

	int option;
	while ((option = getopt(argc, argv, "hgk:mn:st")) != -1) {
		switch (option) {
		case 'h':
			usage();

		case 'g':
			runHostSorts = false;
			break;

		case 'k':
			k = (int)parseInt(10, 0, MAX_ARRAY_LEN, "invalid k-sortedness");
			break;

		case 'm':
			msg("forcing usage of software RNG");
			forceMT = true;
			break;

		case 'n':
			arrayLen = (int)parseInt(10, MIN_ARRAY_LEN, MAX_ARRAY_LEN,
				"invalid unsorted array length");
			break;

		case 's':
			msg("random array shuffling enabled");
			shuffle = true;
			break;

		case 't':
			testSorted = true;
			break;

		case '?': // deal with ill-formed parameterless option
		default:  // ...or invalid option
			usage();
		}
	}
	if (optind < argc) { // reject extra arguments
		fatal("unexpected argument: %s", argv[optind]);
	}
	if (k >= arrayLen) {
		fatal("an array of length %d can't be %d-sorted!", arrayLen, k);
	}

	randInit(forceMT);
	if (!arrayLen) {
		arrayLen = randLen(MIN_RAND_LEN, MAX_RAND_LEN);
	}

	const size_t arraySize = 4 * (size_t)arrayLen;
	//msg("allocating storage...");
	int32_t* const hostUnsortedArray = new int32_t[arrayLen];
	int32_t* const hostSortingArray = new int32_t[arrayLen];
	int32_t* hostReferenceArray = nullptr;

	//msg("%.3f MiB device RAM available, using %.3f MiB", mibibytes(cuMemAvail()),
	//	mibibytes(arraySize));
	int32_t* const devSortingArray = (int32_t*)cuMalloc(arraySize);

	//msg("generating a random array of %d integers...", arrayLen);
	const int radius = randArray(hostUnsortedArray, k, arrayLen, shuffle);

	auto wrapDevRoughsort = [radius](int32_t* const a, const int n) {
		devRoughsort(a, radius, n);
	};

	const bool runDevSorts = true;

	struct {
		const char* name;
		std::function<void(int32_t* const, const int)> sort;
		bool runTest;
		bool onGPU;
	} benchmarks[] = {
		{ "Host Radius ", hostRadius, true, false },
		{ "GPU Radius ", devCheckSortedness, true, false}
	};

	//msg("running sort algorithm benchmarks...");

	for (auto const bench : benchmarks) {
		if (bench.runTest) {
			const bool referenceSort = testSorted && hostReferenceArray == nullptr;

			if (bench.onGPU) {
				cuUpload(devSortingArray, hostUnsortedArray, arraySize);
			}
			else {
				memcpy(hostSortingArray, hostUnsortedArray, arraySize);
			}


			bench.sort(bench.onGPU ? devSortingArray : hostSortingArray, arrayLen);

			auto resultMsg = "";

			if (referenceSort) {
				/* NOTE: We could maintain separate reference and unsorted arrays on the
				GPU, but we probably can't afford the extra device RAM and
				won't need the speedup. */
				hostReferenceArray = new int32_t[arrayLen];
				//msg("copying sorted array to reference for testing further results...");
				if (bench.onGPU) {
					cuDownload(hostReferenceArray, devSortingArray, arraySize);
				}
				else {
					memcpy(hostReferenceArray, hostSortingArray, arraySize);
				}
			}
			if (testSorted) {
				if (bench.onGPU) {
					cuDownload(hostSortingArray, devSortingArray, arraySize);
				}
				if (!testArrayEq(hostSortingArray, hostReferenceArray, arrayLen)) {
					resultMsg = " BUT IS BROKEN";
				}
				bool uniformSrtArr = false;
				bool uniformRefArr = false;
				if (testArrayUniform(hostSortingArray, arrayLen))
				{
					uniformSrtArr = true;
				}
				if (testArrayUniform(hostReferenceArray, arrayLen))
				{
					uniformRefArr = true;
				}
				if (arrayLen < 10 && !uniformRefArr && !uniformSrtArr)
				{
					for (int i = 0; i < arrayLen; i++)
					{
						std::cout << "\t" << i << "," << hostSortingArray[i];
						for (int j = 0; j < arrayLen; j++)
						{
							if (hostReferenceArray[j] == hostSortingArray[i])
							{
								std::cout << "[" << abs(i - j) << "]" << std::endl;
							}
						}
					}
				}
				else if (uniformSrtArr)
				{
					std::cout << "sort array is uniform" << std::endl;
				}
				else if (uniformRefArr)
				{
					std::cout << "ref array is uniform" << std::endl;
				}

			}


			//msg("  %s took %d ms%s", bench.name, ms, resultMsg);
		}
	}

	delete[] hostUnsortedArray;
	delete[] hostSortingArray;
	delete[] hostReferenceArray; // nullptr deletion is safe
	cuFree(devSortingArray);

	return 0;
}