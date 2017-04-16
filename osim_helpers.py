import subprocess


def start_env_server(p=0, ec2=False):
    # subprocess.call(['source', 'deactivate'])
    port = str(5000 + p)
    if ec2:
        server_script_path = '/home/ubuntu/rllab_local/osim_http_server.py'
    else:
        server_script_path = '/Users/anil/Code/rllab/osim_http_server.py'
    
    command = server_script_path + ' -p ' + port
    process = subprocess.Popen(command, shell=True) 
    # subprocess.call(['source', 'activate', 'python3'])
    return process
