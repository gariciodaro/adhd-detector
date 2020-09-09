# base of miniconda image
FROM continuumio/miniconda3

WORKDIR /usr/src/app

# streamlit-specific commands
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'
RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'

# exposing default port for streamlit
EXPOSE 8501

# copying all analysis code to image
COPY . .

#RUN pip install --no-cache-dir -r requirements_pip.txt 
#RUN conda install --file requirements_conda.txt

RUN conda install configparser
RUN conda install pandas
RUN conda install -c anaconda scikit-learn==0.22.1 
RUN conda install -c conda-forge matplotlib==3.1.3 
RUN conda install -c anaconda tornado 
RUN conda install -c conda-forge mne==0.19.2
RUN conda install -c conda-forge xgboost==0.90
RUN conda install -c fastai fastai==1.0.60
RUN conda install -c anaconda seaborn==0.10.0

CMD [ "streamlit", "run", "scripts/app.py" ]