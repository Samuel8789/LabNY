# !/bin/sh

tmux new-session -d -s multiMDL-0
for i in {1..7}; do
tmux split-window
tmux select-layout tiled
done

for a in {0..7}; do
tmux select-pane $a
tmux send-keys -t multiMDL-0.$a "matlab -nodesktop -nosplash -r 'Multicore_Modeling($a)'" C-m
done

tmux attach-session -t multiMDL-0









