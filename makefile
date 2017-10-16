
clean:
	rm ~/TFG/predictions/*.csv
plot:
	gnuplot plot_times.txt
	gnuplot plot_memory.txt
	gnuplot plot_error.txt
R:
	cd ~/TFG/scripts/R/;./run_R.sh

run_py:
	cd ~/TFG/scripts/python/;./run_py.sh

