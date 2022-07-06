#library to check time
import time

#libraries to check resource consumption
import platform
import cpuinfo
import psutil
import os
import threading

cpu_model = cpuinfo.get_cpu_info()["brand_raw"]
ram_total = psutil.virtual_memory()[0]

def clear_console():
    command = 'cls' if os.name in ('nt', 'dos') else 'clear'
    os.system(command)

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

#code for running average frame rate evaluation
#set initial previous time before video capture
frame_count = 0
pTime = 0

#insert code after the current frame has been processed to get current time
cTime = time.time()
frame_count += 1
print(frame_count)
print(frame_count/(cTime - pTime))