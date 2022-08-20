FROM continuumio/miniconda3

LABEL maintainer="Aradhya Purohit"
WORKDIR /tests/

RUN pip install numpy
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install pytest

COPY setup.py setup.py
COPY setup.cfg setup.cfg
COPY src src
COPY tests tests
COPY env.yml env.yml
COPY README.md README.md

RUN pip install -e .
RUN python src/Housing_Price_Prediction/ingest_data.py
RUN python src/Housing_Price_Prediction/train.py
RUN python src/Housing_Price_Prediction/score.py
RUN pytest

EXPOSE 8080

ENTRYPOINT [ "/bin/bash"]
