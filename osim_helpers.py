import subprocess

ec2_python_path = ['',
    '/home/ubuntu/anaconda2/bin',
    '/home/ubuntu/anaconda2/lib/python27.zip',
    '/home/ubuntu/anaconda2/lib/python2.7',
    '/home/ubuntu/anaconda2/lib/python2.7/plat-linux2',
    '/home/ubuntu/anaconda2/lib/python2.7/lib-tk',
    '/home/ubuntu/anaconda2/lib/python2.7/lib-old',
    '/home/ubuntu/anaconda2/lib/python2.7/lib-dynload',
    '/home/ubuntu/anaconda2/lib/python2.7/site-packages',
    '/home/ubuntu/anaconda2/lib/python2.7/site-packages/Sphinx-1.5.1-py2.7.egg',
    '/home/ubuntu/anaconda2/lib/python2.7/site-packages/setuptools-27.2.0-py2.7.egg',
    '/home/ubuntu/anaconda2/lib/python2.7/site-packages/IPython/extensions',
    '/home/ubuntu/.ipython']

local_python_path = ['',
    '/Users/anil/anaconda2/bin',
    '/Users/anil/anaconda2/lib/python27.zip',
    '/Users/anil/anaconda2/lib/python2.7',
    '/Users/anil/anaconda2/lib/python2.7/plat-darwin',
    '/Users/anil/anaconda2/lib/python2.7/plat-mac',
    '/Users/anil/anaconda2/lib/python2.7/plat-mac/lib-scriptpackages',
    '/Users/anil/anaconda2/lib/python2.7/lib-tk',
    '/Users/anil/anaconda2/lib/python2.7/lib-old',
    '/Users/anil/anaconda2/lib/python2.7/lib-dynload',
    '/Users/anil/anaconda2/lib/python2.7/site-packages',
    '/Users/anil/anaconda2/lib/python2.7/site-packages/Sphinx-1.5.1-py2.7.egg',
    '/Users/anil/anaconda2/lib/python2.7/site-packages/aeosa',
    '/Users/anil/anaconda2/lib/python2.7/site-packages/protobuf-3.1.0-py2.7.egg',
    '/Users/anil/anaconda2/lib/python2.7/site-packages/setuptools-27.2.0-py2.7.egg',
    '/Users/anil/anaconda2/lib/python2.7/site-packages/xgboost-0.6-py2.7.egg',
    '/Users/anil/anaconda2/lib/python2.7/site-packages/torchvision-0.1.7-py2.7.egg',
    '/Users/anil/anaconda2/lib/python2.7/site-packages/IPython/extensions',
    '/Users/anil/.ipython']


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
    return process.pid
