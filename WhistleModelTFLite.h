<<<<<<< HEAD
#ifndef WHISTLE_MODEL_TFLITE_H
#define WHISTLE_MODEL_TFLITE_H

#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include <vector>
#include <string>
#include <cassert>

static void error_reporter(void*, const char* format, va_list args) {
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
}

#define MY_ASSERT_NE(v, e) \
    do { \
        if ((v) == (e)) { \
            fprintf(stderr, "%s (%s:%d): Assertion failed!\n", __func__, __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

#define MY_ASSERT_EQ(v, e) \
    do { \
        if ((v) != (e)) { \
            fprintf(stderr, "%s (%s:%d): Assertion failed!\n", __func__, __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

class WhistleModelTFLite
{
public:
    WhistleModelTFLite(const std::string& modelFile);
    ~WhistleModelTFLite();

    // Input a frame(513 float)feature and return the probability
    float predict(const std::vector<float>& inputFeatures);

    // Input 1024-point PCM waveform, automatically extract features and return probability
    float predictFromPCM(const std::vector<float>& samples);

private:
    // Internal FFT feature extraction
    std::vector<float> extractSpectrum(const std::vector<float>& samples);

private:
    const std::string name;
    TfLiteInterpreter* interpreter = nullptr;
    TfLiteDelegate* delegate = nullptr;
    TfLiteTensor* inputTensor = nullptr;
    int numThreads = 2;
};

#endif
=======
#ifndef WHISTLE_MODEL_TFLITE_H
#define WHISTLE_MODEL_TFLITE_H

#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include <vector>
#include <string>
#include <cassert>

static void error_reporter(void*, const char* format, va_list args) {
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
}

#define MY_ASSERT_NE(v, e) \
    do { \
        if ((v) == (e)) { \
            fprintf(stderr, "%s (%s:%d): Assertion failed!\n", __func__, __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

#define MY_ASSERT_EQ(v, e) \
    do { \
        if ((v) != (e)) { \
            fprintf(stderr, "%s (%s:%d): Assertion failed!\n", __func__, __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

class WhistleModelTFLite
{
public:
    WhistleModelTFLite(const std::string& modelFile);
    ~WhistleModelTFLite();

    // Input a frame(513 float)feature and return the probability
    float predict(const std::vector<float>& inputFeatures);

    // Input 1024-point PCM waveform, automatically extract features and return probability
    float predictFromPCM(const std::vector<float>& samples);

private:
    // Internal FFT feature extraction
    std::vector<float> extractSpectrum(const std::vector<float>& samples);

private:
    const std::string name;
    TfLiteInterpreter* interpreter = nullptr;
    TfLiteDelegate* delegate = nullptr;
    TfLiteTensor* inputTensor = nullptr;
    int numThreads = 2;
};

#endif
>>>>>>> b0216cf6b06aa0baadb47cc70d2e87f901352cab
