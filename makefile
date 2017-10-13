clean:
	cd ~/TFG/predictions/
	rm *.csv
plot:
	gnuplot plot_times.txt
	gnuplot plot_memory.txt
	gnuplot plot_error.txt 
	
 

