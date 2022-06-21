frequency=1;%hz
width=0.05;%s
dur=2;%s
outrate=1000
outputv=5
widthsamples=width*outrate
number_of_pulses=frequency*dur;
period=outrate/frequency;
isi=period-widthsamples;


uptimes=zeros(1,number_of_pulses)
 for i=1:number_of_pulses
       uptimes(i)=(i-1)*period
    end
downtimes=uptimes+width*1000





d = daqlist("ni")
d.DeviceInfo(1)
usb_session = daq("ni");
addoutput(usb_session,"Dev1","ao1","Voltage");


signal=create_pulse_train(frequency,width,dur, outrate, outputv);

one_photon_led_pulse(dqo,signal);


function signal=create_pulse_train(frequency,width,dur, outrate, outputv, default_signal )

    assert(width*frequency<=1)
    signal=zeros(dur*outrate,1);
    widthsamples=width*outrate
    number_of_pulses=frequency*dur;
    period=outrate/frequency;
    isi=period-widthsamples;

    for i=1:number_of_pulses
        strt=period*(i-1)+1;
        fin=strt+widthsamples-1;
        signal(strt:fin)=outputv;
    end
    plot(signal)
end

function one_photon_led_pulse(usb_session, signal)
    
    for i=1:length(signal)
        write(usb_session,[signal(i,:)]);
    end
    write(usb_session,[0]);

end



