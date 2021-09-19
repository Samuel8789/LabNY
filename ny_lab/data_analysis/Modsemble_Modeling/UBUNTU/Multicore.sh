# !/bin/sh

echo "Do you wish to set new modeling parameters?"
select yn in "Yes" "No"; do
	case $yn in
		Yes ) matlab -nodesktop -nosplash -r "psetup;Structural_Learning;exit"; break;;
		No ) matlab -nodesktop -nosplash -batch "Structural_Learning;exit"; break;;
	esac
done

echo


./MC.sh;

matlab -nodesktop -nosplash -batch "Multicore_Cleanup;exit";

echo

echo "Multicore Modeling Completed"

echo

echo "Do you wish to conduct core analysis?"
select yn in "Yes" "No"; do
	case $yn in
		Yes ) matlab -nodesktop -nosplash -r "Core_Analysis";break;;
		No ) break;;
	esac
done

echo

exit












