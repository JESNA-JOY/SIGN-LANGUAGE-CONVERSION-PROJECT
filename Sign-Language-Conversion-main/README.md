# asl-recog
LSTM model to recognize ASL gloss

### Data collection
- create venv using 

```powershell
python -m venv asl-recog-venv
```

- Activate virtual environment
    - windows powershell
        ```powershell
        .\asl-recog-venv\Scripts\Activate.ps1
        ```
    - Linux bash
        ```bash
        source ./asl-recog-venv/bin/activate
        ```


- install the requirements using pip

```bash
    pip install -r requirements.txt
```
- Run the cli data generation tool

```bash
python gen_cli.py  -h 
```

- Add the list of gloss in gloss.txt file and invoke the cli program with -r argument

```bash
python gen_cli.py  -r
```

- glosses from gloss.txt can be indexed and generated using -i and -c flag.

    - Eg:-  
    contents in gloss.txt
    ``` 
      0| Hi
      1| Hello 
      2| How are you
      3| Happy birthday
      4| Help me
      5| turn around
      6| see you later
      7| what is your name
    ```

    - c represents the number of gloss in gloss.txt to generate.
       
        ```bash
        python gen_cli.py - c 5
        ```
    this generates data of gloss from 0 to 5
    index 0 -> Hi
    to 
    index 5 -> Turn around
    - i represents the starting index of gloss to generate in gloss.txt to generate.
       
        ```bash
        python gen_cli.py - c 5 -i 2
        ```
    this generates data of gloss from 2 to 5
    where generation starts from  gloss index 2 -> How are you
    to index 5 -> turn around
OR

- provide each gloss name as argument with -g flag

```bash
python gen_cli.py  -g "Happy Birthday"
```

- Also use -v flag to change the number of videos recorded for each gloss

```bash
python gen_cli.py  -v 15
```
- Also use -f flag to change the number of frames in each video

```bash
python gen_cli.py  -f 30
```