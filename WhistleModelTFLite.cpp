<<<<<<< HEAD
#include "WhistleModelTFLite.h"
#include <iostream>
#include <cmath>
#include <fftw3.h>

WhistleModelTFLite::WhistleModelTFLite(const std::string& modelFile)
    : name(modelFile)
{
    TfLiteModel* model = TfLiteModelCreateFromFile(modelFile.c_str());
    MY_ASSERT_NE(model, nullptr);

#ifndef WIN32
    TfLiteXNNPackDelegateOptions xnnOpts = TfLiteXNNPackDelegateOptionsDefault();
    xnnOpts.num_threads = numThreads;
    delegate = TfLiteXNNPackDelegateCreate(&xnnOpts);
    MY_ASSERT_NE(delegate, nullptr);
#endif

    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    MY_ASSERT_NE(options, nullptr);
    TfLiteInterpreterOptionsSetNumThreads(options, numThreads);
#ifndef WIN32
    TfLiteInterpreterOptionsAddDelegate(options, delegate);
#endif

    interpreter = TfLiteInterpreterCreate(model, options);
    MY_ASSERT_NE(interpreter, nullptr);

    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);

    MY_ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);

    inputTensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
    MY_ASSERT_NE(inputTensor, nullptr);
    MY_ASSERT_EQ(TfLiteTensorType(inputTensor), kTfLiteFloat32);

    const TfLiteIntArray* dims = TfLiteTensorDims(inputTensor);
    if (!(dims->size == 3 && dims->data[0] == 1 && dims->data[1] == 513 && dims->data[2] == 1)) {
        std::cerr << "Unexpected input tensor shape." << std::endl;
        exit(1);
    }
}

WhistleModelTFLite::~WhistleModelTFLite()
{
    if (interpreter) {
        TfLiteInterpreterDelete(interpreter);
    }
#ifndef WIN32
    if (delegate) {
        TfLiteXNNPackDelegateDelete(delegate);
    }
#endif
}

float WhistleModelTFLite::predict(const std::vector<float>& inputFeatures)
{
    MY_ASSERT_EQ(inputFeatures.size(), 513);

    float* inputData = TfLiteTensorData(inputTensor);
    for (size_t i = 0; i < 513; ++i) {
        inputData[i] = inputFeatures[i];
    }

    MY_ASSERT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);

    const TfLiteTensor* outputTensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);
    MY_ASSERT_NE(outputTensor, nullptr);

    const float* outputData = TfLiteTensorData(outputTensor);
    return outputData[0];
}

float WhistleModelTFLite::predictFromPCM(const std::vector<float>& samples)
{
    std::vector<float> features = extractSpectrum(samples);
    return predict(features);
}

std::vector<float> WhistleModelTFLite::extractSpectrum(const std::vector<float>& samples)
{
    const int N = 1024;
    const int Nfreq = N/2 + 1;

    if (samples.size() < N) {
        std::cerr << "Error: input too short (expected 1024 samples).\n";
        return {};
    }

    // Hamming window
    std::vector<double> window(N);
    for (int i=0; i<N; ++i)
        window[i] = 0.54 - 0.46 * cos(2.0 * M_PI * i / (N-1));

    // FFT input
    double* in = (double*) fftw_malloc(sizeof(double)*N);
    for (int i=0; i<N; ++i)
        in[i] = samples[i] * window[i];

    // FFT output
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*Nfreq);

    fftw_plan plan = fftw_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);
    fftw_execute(plan);

    // Calculate amplitude
    std::vector<float> logSpectrum(Nfreq);
    for (int i=0; i<Nfreq; ++i) {
        double mag = sqrt(out[i][0]*out[i][0] + out[i][1]*out[i][1]);
        if (mag < 1e-10) mag = 1e-10;
        logSpectrum[i] = static_cast<float>(20.0 * log10(mag));
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return logSpectrum;
}
=======
#include "WhistleModelTFLite.h"
#include <iostream>
#include <cmath>
#include <fftw3.h>

WhistleModelTFLite::WhistleModelTFLite(const std::string& modelFile)
    : name(modelFile)
{
    TfLiteModel* model = TfLiteModelCreateFromFile(modelFile.c_str());
    MY_ASSERT_NE(model, nullptr);

#ifndef WIN32
    TfLiteXNNPackDelegateOptions xnnOpts = TfLiteXNNPackDelegateOptionsDefault();
    xnnOpts.num_threads = numThreads;
    delegate = TfLiteXNNPackDelegateCreate(&xnnOpts);
    MY_ASSERT_NE(delegate, nullptr);
#endif

    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    MY_ASSERT_NE(options, nullptr);
    TfLiteInterpreterOptionsSetNumThreads(options, numThreads);
#ifndef WIN32
    TfLiteInterpreterOptionsAddDelegate(options, delegate);
#endif

    interpreter = TfLiteInterpreterCreate(model, options);
    MY_ASSERT_NE(interpreter, nullptr);

    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);

    MY_ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);

    inputTensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
    MY_ASSERT_NE(inputTensor, nullptr);
    MY_ASSERT_EQ(TfLiteTensorType(inputTensor), kTfLiteFloat32);

    const TfLiteIntArray* dims = TfLiteTensorDims(inputTensor);
    if (!(dims->size == 3 && dims->data[0] == 1 && dims->data[1] == 513 && dims->data[2] == 1)) {
        std::cerr << "Unexpected input tensor shape." << std::endl;
        exit(1);
    }
}

WhistleModelTFLite::~WhistleModelTFLite()
{
    if (interpreter) {
        TfLiteInterpreterDelete(interpreter);
    }
#ifndef WIN32
    if (delegate) {
        TfLiteXNNPackDelegateDelete(delegate);
    }
#endif
}

float WhistleModelTFLite::predict(const std::vector<float>& inputFeatures)
{
    MY_ASSERT_EQ(inputFeatures.size(), 513);

    float* inputData = TfLiteTensorData(inputTensor);
    for (size_t i = 0; i < 513; ++i) {
        inputData[i] = inputFeatures[i];
    }

    MY_ASSERT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);

    const TfLiteTensor* outputTensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);
    MY_ASSERT_NE(outputTensor, nullptr);

    const float* outputData = TfLiteTensorData(outputTensor);
    return outputData[0];
}

float WhistleModelTFLite::predictFromPCM(const std::vector<float>& samples)
{
    std::vector<float> features = extractSpectrum(samples);
    return predict(features);
}

std::vector<float> WhistleModelTFLite::extractSpectrum(const std::vector<float>& samples)
{
    const int N = 1024;
    const int Nfreq = N/2 + 1;

    if (samples.size() < N) {
        std::cerr << "Error: input too short (expected 1024 samples).\n";
        return {};
    }

    // Hamming window
    std::vector<double> window(N);
    for (int i=0; i<N; ++i)
        window[i] = 0.54 - 0.46 * cos(2.0 * M_PI * i / (N-1));

    // FFT input
    double* in = (double*) fftw_malloc(sizeof(double)*N);
    for (int i=0; i<N; ++i)
        in[i] = samples[i] * window[i];

    // FFT output
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*Nfreq);

    fftw_plan plan = fftw_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);
    fftw_execute(plan);

    // Calculate amplitude
    std::vector<float> logSpectrum(Nfreq);
    for (int i=0; i<Nfreq; ++i) {
        double mag = sqrt(out[i][0]*out[i][0] + out[i][1]*out[i][1]);
        if (mag < 1e-10) mag = 1e-10;
        logSpectrum[i] = static_cast<float>(20.0 * log10(mag));
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return logSpectrum;
}
>>>>>>> b0216cf6b06aa0baadb47cc70d2e87f901352cab
