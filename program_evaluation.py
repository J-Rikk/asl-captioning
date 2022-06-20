#library to check time
import time

#libraries to check resource consumption
import platform
import cpuinfo
import psutil
import os
import threading

#retrieves device's CPU details and total virtual memory
cpu_model = cpuinfo.get_cpu_info()["brand_raw"]
ram_total = psutil.virtual_memory()[0]

#function that checks the OS then clears the console 
def clear_console():
    command = 'cls' if os.name in ('nt', 'dos') else 'clear'
    os.system(command)

#logs and reports CPU and RAM usage every second; data is saved in separate text files
def usage_report(cpu_model, ram_total):
    open('cpu_log.txt','w').close()
    open('ram_log.txt','w').close()

    while True:
        cpu_log = open('cpu_log.txt','a')
        ram_log = open('ram_log.txt','a')

        time.sleep(1)

        ram_percent = psutil.virtual_memory()[2]
        ram_used = psutil.virtual_memory()[3]

        cpu_log.write(f'{psutil.cpu_percent()}\n')
        ram_log.write(f'{round(ram_used/10**9, 2)}\n')

        clear_console()

        print(f'CPU Usage: {psutil.cpu_percent()}% ({cpu_model})')
        print(f'RAM Usage: {ram_percent}% ({round(ram_used/10**9, 2)}/{round(ram_total/10**9,2)} GB)')

        cpu_log.close()
        ram_log.close()

t1 = threading.Thread(target=usage_report, args=[cpu_model, ram_total])
t1.start()

#code for frame rate evaluation
#set initial previous time before video capture
pTime = 0

#insert code after the current frame has been processed to get current time
cTime = time.time()
fps = 1 / (cTime - pTime)
#changes previous time to current time for next frame
pTime = cTime