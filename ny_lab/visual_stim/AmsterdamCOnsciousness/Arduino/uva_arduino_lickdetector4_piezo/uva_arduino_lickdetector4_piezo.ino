
#include "QuickStats.h" // quickstats is used for computing quick statistics
// Arduino script for an arduino-based dual lick detector that works with piezosensors
// MOL 2017: Made script

QuickStats stats; // Instance of the QuickStats library

const int PIEZO_PIN1 = A1; // Piezo output
const int PIEZO_PIN2 = A2; // Piezo output
float piezoV1;
float piezoV2;

int rewardTime1 = 200; // Initialize value reward time for port 1 i.e. right side
int rewardTime2 = 200; // Initialize value reward time for port 2 i.e. left side
unsigned int iBuffer = 0; // index variable
float thres = 1; // threshold for capactititave sensor lick detection (in Volts)
int reward1 = 10; //output pin for reward port 1
int reward2 = 11; // output pin for reward port 2
float sensorValue1; // value at sensor 1
float sensorValue2; // value at sensor 2
int detectedSensor = 0; // value for storing the detected sensor
int baselineSamplePeriod = 50; // sampling period for baseline in ms
unsigned long lastBaseSampleTime = 0; // time of last baseline sample
float baseline1[40]; // vector containing baseline values for sensor 1
float baseline2[40]; // vector containing baseline values for sensor 2
float buf1[20]; // vector for buffering the sensor values, i.e. for smoothing the signal
float buf2[20]; // vector for buffering the sensor values, i.e. for smoothing the signal
float medBase1 = 0; // median of baseline sensor 1
float medBase2 = 0; // median of baseline sensor 2
unsigned int iBaseline = 0; // iterator for baseline sampling
unsigned long runtime = 0;
unsigned long testtime = 0;
unsigned long rewardtime = 0;
unsigned long lastlick = 0;
boolean bSuccRight = 1;
boolean bSuccLeft = 1;
int LEDpin = 12;
unsigned long baudrate = 115200; // baudrate, should be the same as reader (i.e. matlab)
char Ctrl = 'A'; // Var for storing serial connection character

void setup()
{
  Serial.begin(baudrate);
  pinMode(reward1, OUTPUT);  //fluid output (tobias thinks right side)
  pinMode(reward2, OUTPUT);  //fluid output (tobias thinks left side)
  pinMode(12, OUTPUT); //ttl pulse for LED
  digitalWrite(reward1, LOW);
  digitalWrite(reward2, LOW);
  digitalWrite(LEDpin, LOW);
  // fill baseline with default values
  for (int i = 0; i < 40; i++) {
    baseline1[i] = 200000;
    baseline2[i] = 200000;
  }
}

// This is the actual loop that the arduino continuously runs through
void loop()
{
  runtime = millis(); // readin the current runtime in milliseconds
  // Check sensors from 10 ms after last lick
  if (runtime - lastlick > 10) {
    checkSensors();
  }
  // get a baselinesample if period has passed
  if (runtime - lastBaseSampleTime > baselineSamplePeriod) {
    // Reset timer
    lastBaseSampleTime = runtime;
    baseline1[iBaseline] = analogRead(PIEZO_PIN1) / 1023.0 * 5.0;
    baseline2[iBaseline] = analogRead(PIEZO_PIN2) / 1023.0 * 5.0;

    // Get the median of the baseline
    medBase1 = stats.median(baseline1, 40);
    medBase2 = stats.median(baseline2, 40);

    // increment the iterator
    iBaseline++;
    if (iBaseline > 39) {
      iBaseline = 0;
    }
  }
  detectedSensor = 0;
  iBuffer += 1;
  if (iBuffer > 19) {
    iBuffer = 0;
  }
} // end loop

// Additional functions:
void openOne(int half) {
  digitalWrite(reward1, HIGH);
  delay(rewardTime1);
  digitalWrite(reward1, LOW);
}

void openTwo(int half) {
  digitalWrite(reward2, HIGH);
  delay(rewardTime2);
  digitalWrite(reward2, LOW);
}

// serialEvent is run every loop whenever there is input from the serial connection
// If this is the case, check the serial connection
void serialEvent()
{
  checkSerial();
}

// Check the serial connection for commands
void checkSerial() {
  if (Serial.find("I")) {
    delay(1);
    // Read in the serial message
    Ctrl = Serial.read();
    switch ( Ctrl ) {
    case 'P':
      Ctrl = Serial.read();
      // If right side reward is requested open valve 1
      Serial.println('D'); // send a 'D' to say it worked
      if (Ctrl == 'R') {
        rewardTime1 = Serial.parseInt();
        openOne(1);
      }
      // If left side reward is requested open valve 2
      else if (Ctrl == 'L') {
        rewardTime2 = Serial.parseInt();
        openTwo(1);
      }
      break;

      // Set the threshold for lick detection to value
    case 'F':
      thres = Serial.parseInt();
      Serial.println('D');
      break;
      // Send current runtime over serial port
    case 'C':
      Serial.println('R');
      Serial.println(runtime);
      break;
    default:
      ;
    }
  }
}

void checkSensors() {

  // Check the sensors
  buf1[iBuffer] = abs(analogRead(PIEZO_PIN1) / 1023.0 * 5.0 - medBase1);
  buf2[iBuffer] = abs(analogRead(PIEZO_PIN2) / 1023.0 * 5.0 - medBase2);

  sensorValue1 = 0;
  sensorValue2 = 0;

  // iterate over the buffers and update the sensor values %Take 20 samples
  for (int i = 0; i < 20; i++) {
    sensorValue1 += buf1[i];
    sensorValue2 += buf2[i];
  }

//  This part is to view the value of one sensor and the threshold and see whether the threshold is good:
  Serial.print(sensorValue1); // Print the voltage.
  Serial.print(",");          //seperator
  Serial.print(sensorValue2); // Print the voltage.
  Serial.print(",");          //seperator
  Serial.println(thres);      // Print the threshold.

  // Check if sensor 1 crossed threshold
  if (sensorValue1 > thres) {
    detectedSensor = 1;
    if (bSuccRight) {
      bSuccRight = false;
      lastlick = runtime;
//      printRight();
    }
  }
  else { //reset again if value goes below threshold
    bSuccRight = true;
  }

  // Check if sensor 2 crossed threshold
  if (sensorValue2 > thres) {
    detectedSensor = 2;
    if (bSuccLeft) {
      bSuccLeft = false;
      lastlick = runtime;
//      printLeft();
    }
  }
  else { //reset again if value goes below threshold
    bSuccLeft = true;
  }
}

// Tobias
void printRight() {
  Serial.println('O');
  Serial.println("Y");
  Serial.println(runtime);
}

// Tobias
void printLeft() {
  Serial.println('O');
  Serial.println("Z");
  Serial.println(runtime);
}

