
#include "CapacitiveSensor.h" // The capacitativesensor.h is used, this should be in the folder
// uva-arduinolickdetector1
// Arduino script for an arduino-based dual lick detector
// test
// Sven: A lot of functions I don't use, such as reaction time, enabling
// ports etc. Could be taken out in future versions
CapacitiveSensor   cs_4_2 = CapacitiveSensor(4,2);        // 10M resistor between pins 4 & 2, pin 2 is sensor pin, add a wire and or foil if desired
CapacitiveSensor   cs_4_6 = CapacitiveSensor(4,6);        // 10M resistor between pins 4 & 6, pin 6 is sensor pin, add a wire and or foil


int rewardTime1 = 200; // reward time for port 1 i.e. right side
int rewardTime2 = 200; // reward time for port 2 i.e. left side
unsigned int indx=0; // index variable
int thres = 10000; // threshold for capactititave sensor lick detection
int reward1 = 10; //output pin for reward port 1
int reward2 = 11; // output pin for reward port 2
long sensorValue1; // value at sensor 1
long sensorValue2; // value at sensor 2
int detectedSensor = 0; // value for storing the detected sensor
float threshold1 = 200; // sven edit
float threshold2 = 200; // sven edit
long buf1[20];
long buf2[20];
int gavepassive = 0;
unsigned long runtime = 0;
unsigned long testTime = 0;
int easymode;
int wentthrough = 0;
unsigned long rewardtime = 0;
int rewardduration = 0;
unsigned long lastlick = 0;
boolean bSucc = 1;
unsigned long trialonset2 = 0;
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
      // Turn easy mode on (1) or off (0)
      case 'M':
        easymode = Serial.parseInt();
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
//          Serial.println(threshold1);
//          Serial.println(threshold2);
//}


void checkSensors(){
  // Check the sensors
  buf1[indx] = cs_4_2.capacitiveSensorRaw(1);
  buf2[indx] = cs_4_6.capacitiveSensorRaw(1);

  sensorValue1 = 0;
  sensorValue2 = 0;

  // iterate over the buffers and update the sensor values
  for(int i= 0; i < 20; i++){
    sensorValue1 += buf1[i];
    sensorValue2 += buf2[i];
  }
  // Check if sensor 1 crossed threshold
  if(sensorValue1 - threshold1 > thres){
    detectedSensor = 1;
    if(bSucc){
      bSucc = false;
      lastlick = runtime;
      printRight();
    }
  }
  // Check if sensor 2 crossed threshold
  else if(sensorValue2 - threshold2 > thres){
    detectedSensor = 2;
    if(bSucc){
      bSucc = false;
      lastlick = runtime;
      printLeft();
    }
  }
  else if(sensorValue2 - threshold2 < thres * 0.5 &&  sensorValue1 - threshold1 < thres * 0.5)
  {
    bSucc = true;
  }

  // }
  // if(runtime - Trialtime < 3000 && runtime - Trialtime > 50 && Enable > 0 && detectedSensor > 0){
  //   if(Enable == 1 && detectedSensor == 1){
  //       openOne(0);
  //       wentthrough = sensorValue1 - threshold1;
  //       printFeedback('1');
  //       Enable = 0;
  //     }
  //     else if(Enable == 2 && detectedSensor == 2 ){
  //       openTwo(0);
  //       wentthrough = sensorValue2 - threshold2;
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
  //        wentthrough = max(sensorValue1 - threshold1, sensorValue2 - threshold2);
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
  if (runtime - lastlick > 10){
    checkSensors();
    // This is to keep the threshold1 and threshold 2 close to the values of the sensor
    if(detectedSensor == 0){
       if(sensorValue1 < threshold1*10) threshold1 = (threshold1*9 + sensorValue1)/10;
       if(sensorValue2 < threshold2*10) threshold2 = (threshold2*9 + sensorValue2)/10;
       // Sven added this to also have an increase in the thresholds
       //if(sensorValue1 > threshold1) threshold1 = threshold1 + 0.001;
       //if(sensorValue2 > threshold2) threshold2 = threshold2 + 0.001;
    }
  }

  detectedSensor = 0;
  indx += 1;
  if(indx > 19) {
    indx = 0;
  }

} // end loop

// serialEvent is run every loop whenever there is input from the serial connection
// If this is the case, check the serial connection
void serialEvent()
{
  checkSerial();
}
