// ServoControl 
// By Reinder Dorman Okt 2017, UvA
//
// Runs on the arduino to turn the servo to a particular position in fron the animal. 
// Waits for a serialcode about the position.
// input is the angle of the servo. in the mouserig be sure to have a correct angle.
// i.e. we need to set a nul-point which is at the animals face. From here on the servo
// may ONLY turn away, as to not hit the animal on the head.
//
// todo: find a way to make servos go slower, FIX THE FUCKING SECOND MONITOR AND DEADKEYS ON THIS GODAWEFUL PCAS#$(%U$


#include <VarSpeedServo.h> 
 
  VarSpeedServo myservo;  // create servo object to control a servo 
                  // a maximum of eight servo objects can be created 
 
  int serialInput;    // variable to store the servo position 
  
  
 
void setup() 
{ 
  myservo.attach(9);  // attaches the servo on pin 9 to the servo object 
  Serial.begin(9600);
} 
 
 
void loop() 
{
  while(Serial.available() <= 0)
  {
    
  }
    int serialInput = Serial.read(); 
    myservo.write(serialInput,20);
    delay(15);
    myservo.wait();
    Serial.write(1);
    
} 
