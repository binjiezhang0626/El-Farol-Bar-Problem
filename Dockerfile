FROM pklehre/niso2019-lab5
ADD bxz858.py /bin

RUN apt-get update
RUN apt-get -y install python-numpy
RUN apt-get -y install python-argparse

RUN ["/bin/bash", "-c", "cp /bin/app_niso2019_lab5 /bin/app_niso_lab5"]

CMD ["-username", "bxz858", "-submission", "python3 /bin/bxz858.py"]