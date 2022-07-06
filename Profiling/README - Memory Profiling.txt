(0) Install libraries using pip.
	pip install memory_profiler
	pip install pyqt5
(1) From terminal run "mnist_mem_profiling.py" using the command below
	python -m memory_profiler mnist_mem_profiling.py > mnist_mem_profile.txt
(2) From terminal run "mnist_mem_profiling.py" again and then plot the batch file. Commands below.
	mprof run mnist_mem_profiling.py
	mprof plot
(3) A window with the plot will open. Save the plot as "mnist_mem_plot.png" using the save button in the toolbar.