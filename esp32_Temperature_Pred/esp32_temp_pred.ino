/* Copyright 2024 Chirale, TensorFlow Authors. All Rights Reserved.

This sketch is derived from the classic Hello World example of the general 
TensorFlow Lite Micro library. It has been adapted and simplified by Chirale 
to conform to the typical style of Arduino sketches. 
It has been tested on an Arduino Nano 33 BLE.
The sketch implements a Deep Neural Network pre-trained on calculating 
the function sin(x). 
By sending a value between 0 and 2*Pi via the Serial Monitor, 
both the value inferred by the DNN model and the actual value 
calculated using the Arduino math library are displayed.

It shows how to use TensorFlow Lite Library on Arduino.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// include main library header file
#include <Chirale_TensorFlowLite.h>

// include static array definition of pre-trained model
#include "model.h"

// This TensorFlow Lite Micro Library for Arduino is not similar to standard
// Arduino libraries. These additional header files must be included.
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Globals pointers, used to address TensorFlow Lite components.
// Pointers are not usual in Arduino sketches, future versions of
// the library may change this...
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// There is no way to calculate this parameter
// the value is usually determined by trial and errors
// It is the dimension of the memory area used by the TFLite interpreter
// to store tensors and intermediate results
constexpr int kTensorArenaSize = 2000;

// Keep aligned to 16 bytes for CMSIS (Cortex Microcontroller Software Interface Standard)
// alignas(16) directive is used to specify that the array 
// should be stored in memory at an address that is a multiple of 16.
alignas(16) uint8_t tensor_arena[kTensorArenaSize];


void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("Starting TinyML...");

  model = tflite::GetModel(model_tflite);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Wrong TensorFlow Lite schema version!");
    while (true);
  }

  static tflite::AllOpsResolver resolver;

  static tflite::MicroInterpreter static_interpreter(
      model,
      resolver,
      tensor_arena,
      kTensorArenaSize
  );

  interpreter = &static_interpreter;

  TfLiteStatus status = interpreter->AllocateTensors();

  if (status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed!");
    while (true);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("TensorFlow initialization OK!");
}


void loop() {

  float temperatures[10] = {
    25.1, 25.3, 25.6, 25.8, 26.0,
    26.1, 26.3, 26.4, 26.6, 26.8
  };

  for (int i = 0; i < 10; i++) {
    input->data.f[i] = temperatures[i];
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return;
  }

  float prediction = output->data.f[0];

  Serial.print("Predicted next temperature: ");
  Serial.print(prediction);
  Serial.println(" °C");

  delay(1000);
}
