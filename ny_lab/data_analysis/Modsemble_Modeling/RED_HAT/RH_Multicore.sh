# !/bin/sh

 
matlab -nodesktop -nosplash -batch "RH_Structural_Learning;exit"; break;;

./MC.sh;

matlab -nodesktop -nosplash -batch "Multicore_Cleanup;exit";

echo

echo "Multicore Modeling Completed"

echo

exit










