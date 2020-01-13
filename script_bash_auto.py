import os
import psutil


def get_temp_cpu_ind():
    temp = psutil.sensors_temperatures()
    temp = str(temp)
    ind=0
    for i in range(0,len(temp)-1):
        if temp[i]=="c" and temp[i+1]=="u" and temp[i+2]:
            return ind
        ind = ind+1
    return None

def get_temp_cpu():
    ind = get_temp_cpu_ind()
    temp = psutil.sensors_temperatures()
    temp = str(temp)
    return float(temp[ind+8:ind+10])




def create_var_env(command):
    command = command.split(";")
    for i in command:
        os.system(i)
    os.system("ls")
    print(command)


def get_var_env():
    vars = os.getenv('USERNAME')
    return os.environ,vars

if __name__=='__main__':
    create_var_env("ls;ifconfig;chmod")
    commande = "CharCPU = top -n 1 -b |grep root | cut -c42-46 |paste -sd+ |bc"
    create_var_env(commande)
    environ,vars = get_var_env()
    print("USERNAME : ",vars)
    print(environ)
    print(0)
    print(get_temp_cpu())
