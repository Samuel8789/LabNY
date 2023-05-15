
#include "CapacitiveSensor.h" // The capacitativesensor.h is used, this should be in the folder, also the .cpp file
#include "QuickStats.h" // quickstats is used for computing quick statistics
// uva-arduinolickdetector2
// Arduino script for an arduino-based dual lick detector
// test
// Sven: A lot of functions I don't use, such as reaction time, enabling
// ports etc. Could be taken out in future versions
CapacitiveSensor   cs_4_2 = CapacitiveSensor(4,2);        // 10M resistor between pins 4 & 2, pin 2 is sensor pin, add a wire and or foil if desired
CapacitiveSensor   cs_4_6 = CapacitiveSensor(4,6);        // 10M resistor between pins 4 & 6, pin 6 is sensor pin, add a wire and or foil
QuickStats stats; // Instance of the QuickStats library

int rewardTime1 = 200; // reward time for port 1 i.e. right side
int rewardTime2 = 200; // reward time for port 2 i.e. left side
unsigned int iBuffer=0; // index variable
float thres = 150; // threshold for capactititave sensor lick detection (i.e. x times the baseline)
int reward1 = 10; //output pin for reward port 1
int reward2 = 11; // output pin for reward port 2
float sensorValue1; // value at sensor 1
float sensorValue2; // value at sensor 2
int detectedSensor = 0; // value for storing the detected sensor
int baselineSamplePeriod = 100; // sampling period for baseline in ms
unsigned long lastBaseSampleTime = 0; // time of last baseline sample
float baseline1[40]; // vector containing baseline values for sensor 1
float baseline2[40]; // vector containing baseline values for sensor 2
float buf1[20]; // vector for buffering the sensor values, i.e. for smoothing the signal
float buf2[20]; // vector for buffering the sensor values, i.e. for smoothing the signal
float medBase1 = 20000; // median of baseline sensor 1
float medBase2 = 20000; // median of baseline sensor 2
unsigned int iBaseline = 0; // iterator for baseline sampling
unsigned long runtime = 0;
unsigned long testtime = 0;
unsigned long rewardtime = 0;
unsigned long lastlick = 0;
boolean bSucc = 1;
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
  for(int i= 0; i < 40; i++){
    baseline1[i] = 200000;
    baseline2[i] = 200000;
  }
}

void openOne(int half){
  digitalWrite(reward1, HIGH);
  delay(rewardTime1);
  digitalWrite(reward1, LOW);
}

void openTwo(int half){
  digitalWrite(reward2, HIGH);
  delay(rewardTime2);
  digitalWrite(reward2, LOW);
}
// Check the serial connection for commands
void checkSerial(){
  if(Serial.find("I")){
    delay(1);
    // Read in the serial message
    Ctrl = Serial.read();
    switch( Ctrl ) {
      // Dont know what this is for yet
      case 'Q':
        int LEDval;
        LEDval = Serial.parseInt();
        if(LEDval){
          digitalWrite(LEDpin, HIGH);
        }
        else{
          digitalWrite(LEDpin, LOW);
        }
        break;
      // Set reward duration for left port
      case 'L':
        rewardTime1 = Serial.parseInt();
        Serial.println('D');
        break;
      // Set reward duration for right port
      case 'R':
        rewardTime2 = Serial.parseInt();
        Serial.println('D');
        break;
      // Give a reward on side provided by char in next line
      case 'P':
        Ctrl = Serial.read();
        // If right side reward is requested open valve 2
        if (Ctrl == 'R'){
          openTwo(1);
        }
        // If left side reward is requested open valve 1
        else if(Ctrl == 'L') {
          openOne(1);
        }
        Serial.println('D'); // send a 'D' to say it worked
        break;
      // Give a passive reward
      //            case 'P':
      //                if(Enable == 1 && runtime-trialonset > Timeout){
      //                  openOne(1);
      //                  gavepassive = 1;
      //                }
      //                else if(Enable == 2 && runtime-trialonset > Timeout){
      //                  openTwo(1);
      //                  gavepassive = 1;
      //                }
      //                break;
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


// Print feedback over serial connection
//void printFeedback(char m){
//          Serial.println('O');
//          Serial.println('X');
//          Serial.println(m);
//          Serial.println(runtime - Trialtime);
//          Serial.println(gavepassive);
//          Serial.println(wentthrough);
//          Serial.println(baseline1);
//          Serial.println(baseline2);
//}


void checkSensors(){
  // Check the sensors
  buf1[iBuffer] = cs_4_2.capacitiveSensorRaw(1);
  buf2[iBuffer] = cs_4_6.capacitiveSensorRaw(1);

  sensorValue1 = 0;
  sensorValue2 = 0;

  // iterate over the buffers and update the sensor values
  for(int i= 0; i < 20; i++){
    sensorValue1 += buf1[i];
    sensorValue2 += buf2[i];
  }
  // Check if sensor 1 crossed threshold
  if(sensorValue1 / medBase1 > thres/100){
    detectedSensor = 1;
    if(bSucc){
      bSucc = false;
      lastlick = runtime;
      printRight();
    }
  }
  // Check if sensor 2 crossed threshold
  else if(sensorValue2 / medBase2 > thres/100){
    detectedSensor = 2;
    if(bSucc){
      bSucc = false;
      lastlick = runtime;
      printLeft();
    }
  }
  // Check if both sensors are close to baseline and reset detection
  else if(sensorValue2 < 1.1 * medBase2 &&  sensorValue1 < 1. * medBase1)
  {
    bSucc = true;
  }
  
  // }
  // if(runtime - Trialtime < 3000 && runtime - Trialtime > 50 && Enable > 0 && detectedSensor > 0){
  //   if(Enable == 1 && detectedSensor == 1){
  //       openOne(0);
  //       wentthrough = sensorValue1 - baseline1;
  //       printFeedback('1');
  //       Enable = 0;
  //     }
  //     else if(Enable == 2 && detectedSensor == 2 ){
  //       openTwo(0);
  //       wentthrough = sensorValue2 - baseline2;
  //       printFeedback('2');
  //       Enable = 0;
  //     }
  //     else if (Enable > 0){
  //        if(easymode){
  //          switch(Enable){
  //            case 1:
  //              openOne(1);
  //              break;
  //            case 2:
  //              openTwo(1);
  //              break;
  //          }
  //         }
  //        wentthrough = max(sensorValue1 - baseline1, sensorValue2 - baseline2);
  //        printFeedback('0');
  //        Enable = 0;
  //     }
  //  }
}

// Tobias
void printRight(){
  Serial.println('O');
  Serial.println("Y");
  Serial.println(runtime);
}

// Tobias
void printLeft(){
  Serial.println('O');
  Serial.println("Z");
  Serial.println(runtime);
}

// This is the actual loop that the arduino continuously runs through
void loop()
{
  runtime = millis(); // readin the current runtime in milliseconds
  // Check sensors from 10 ms after last lick
  if (runtime - lastlick > 10){
    checkSensors();    
  }
  // get a baselinesample if period has passed
  if (runtime - lastBaseSampleTime > baselineSamplePeriod) {
    // Reset timer
    lastBaseSampleTime = runtime; 
    baseline1[iBaseline] = sensorValue1;
    baseline2[iBaseline] = sensorValue2;
    // Get the median of the baseline
    medBase1 = stats.median(baseline1,1);
    medBase2 = stats.median(baseline2,1);
    // increment the iterator
    iBaseline++;
    if(iBaseline > 39) {
      iBaseline = 0;
    }
  }
  detectedSensor = 0;
  iBuffer += 1;
  if(iBuffer > 19) {
    iBuffer = 0;
  }
} // end loop

// serialEvent is run every loop whenever there is input from the serial connection
// If this is the case, check the serial connection
void serialEvent()
{
  checkSerial();
}
