# Copy Openfold3 folder and build dependencies needed for testing
FROM openfoldconsortium/openfold3:stable

COPY openfold3 /opt/openfold3
COPY environments/development.txt /opt/
COPY setup.py /opt/

WORKDIR /opt/
RUN pip install -r development.txt
RUN python setup.py install

WORKDIR /opt/openfold3