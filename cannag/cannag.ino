#include <Arduino.h>
#include "HX711.h"

const int LOADCELL_DOUT_PIN = 2;
const int LOADCELL_SCK_PIN = 3;
const long THRESHOLD = 1000;  // Ngưỡng để thông báo

HX711 scale;

void setup() {
  Serial.begin(57600);
  Serial.println("HX711 Demo");
  Serial.println("Initializing the scale");

  scale.begin(LOADCELL_DOUT_PIN, LOADCELL_SCK_PIN);
  scale.set_scale(1132);
  // Khởi tạo giá trị ban đầu
  scale.tare();  // reset the scale to 0

}

void loop() {
  Serial.println("Vui lòng đặt vật nặng lên cân!");
  Serial.print("Khoi luong cua vat: ");
  Serial.println(scale.get_units(10), 5);

  delay(5000);
}
