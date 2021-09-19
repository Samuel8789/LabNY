# !/bin/sh

echo "Do you want to install beta"
select yn in "Yes" "No"; do
	case $yn in
		Yes ) matlab -nodesktop -nosplash -batch "RHsetup;exit"; break;; 
		No ) exit
	esac
done

echo "Designated MATLAB MEX Directory"

make

echo "Installing Terminal Multiplexer (TMUX)"

sudo yum install tmux

echo "Installation Complete"




