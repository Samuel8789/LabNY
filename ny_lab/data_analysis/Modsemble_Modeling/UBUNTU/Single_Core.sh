# !/bin/sh

echo "Do you wish to set new modeling parameters?"
select yn in "Yes" "No"; do
	case $yn in
		Yes ) matlab -nodesktop -nosplash -r "psetup;Structural_Learning;Single_Core_PE;exit"; break;;
		No ) matlab -nodesktop -nosplash -batch "Structural_Learning;Single_Core_PE;exit"; break;;
	esac
done

echo

echo "Modeling completed. Do you wish to conduct core analysis?"
select yn in "Yes" "No"; do
	case $yn in
		Yes ) matlab -nodesktop -nosplash -r "Core_Analysis";break;;
		No ) break;;
	esac
done

echo

exit
