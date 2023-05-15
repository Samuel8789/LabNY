//PiezoController
//Script for arduino-based control of piezos
int signalPin = 11;
int gateLeftPin = 5;
int gateRightPin = 6;
unsigned long baudrate = 115200; //should be the same as reader
int Intensity = 0; //0-100
int DutyCycle = 0;
float Decr=0;
char Ctrl = 'A';

void setup()
{
  Serial.begin(baudrate);
  pinMode(signalPin,OUTPUT); //Control for driver
  pinMode(gateLeftPin,OUTPUT); //Gate transistor LEFT
  pinMode(gateRightPin,OUTPUT); //Gate transistor RIGHT 
  digitalWrite(signalPin, LOW);
  digitalWrite(gateLeftPin, LOW);
  digitalWrite(gateRightPin, LOW);
}

void movepiezo(int Intensity){
  DutyCycle = Intensity*255/100;
  DutyCycle = constrain(DutyCycle,0,255);
  analogWrite(signalPin, DutyCycle);
  delay(360); //5*TAU with TAU=72ms
  //linear decrease produces sound   
  analogWrite(signalPin,0); 
}

void checkSerial(){
  if (Serial.find("I")){
    delay(1);
    // Read in the serial message
    Ctrl = Serial.read();
    switch ( Ctrl ) {
      case 'V':
      Serial.println('P'); //P or piezo, L for lickdetector
      break;
      case 'P':
      Serial.println('D'); // send a 'D' to say it worked
      Ctrl = Serial.read();
      //move message : IP R/L Intensity f
      if (Ctrl == 'L'){
          digitalWrite(gateLeftPin, HIGH);
          Intensity = Serial.parseInt(); 
          movepiezo(Intensity);     
          digitalWrite(gateLeftPin, LOW);
      }  
      else if (Ctrl == 'R') {
          digitalWrite(gateRightPin, HIGH);
          Intensity = Serial.parseInt();
          movepiezo(Intensity);
          digitalWrite(gateRightPin, LOW);
      }
      break;
    }
  }
}

void loop()
{
  //runtime = millis();
}

// serialEvent is run every loop whenever there is input from the serial connection
// If this is the case, check the serial connection
void serialEvent()
{
  checkSerial();
}

