# !/bin/sh


matlab -nodesktop -nosplash -batch "Struc_Sam;Single_Core_PE;exit";

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
