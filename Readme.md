# Ephys Tools
This repository builds on existing tools and extends funcionality according to user specific needs.
The basis for the tool box uses the [*neo*](https://github.com/NeuralEnsemble/python-neo) package and will incorporate functions from the [NWB](https://github.com/NeurodataWithoutBorders/pynwb) implementation in python.
Additionally, the tool box relies on analysis standards from literature and implementations of analysis and convenient functions from the [schmitzlab](https://github.com/Schmitz-lab).

Further documentation can be found at the [documentation page](https://danielparthier.github.io/ephystools-docs/).

When contributing code make sure to follow the style guidelines and check with code formatter *black* (can be istalled via `pip install black`)

```bash
$ black ephys/classes/class_functions.py 

All done! ✨ 🍰 ✨
1 file left unchanged.
```

To keep the code tidy and assure consistent standards please use *pylint* (can be installed via `pip install pylint`). By running *pylint* potential drop in quality can be detected and fixed before commiting code.

```bash
$ python -m pylint ephys/classes/class_functions.py 

************* Module ephys.classes.class_functions
ephys/classes/class_functions.py:130:0: C0301: Line too long (102/100) (line-too-long)
ephys/classes/class_functions.py:35:30: W1401: Anomalous backslash in string: '\('. String constant might be missing an r prefix. (anomalous-backslash-in-string)
ephys/classes/class_functions.py:35:34: W1401: Anomalous backslash in string: '\)'. String constant might be missing an r prefix. (anomalous-backslash-in-string)
ephys/classes/class_functions.py:33:8: C0103: Variable name "Analogsignals" doesn't conform to snake_case naming style (invalid-name)

------------------------------------------------------------------
Your code has been rated at 9.43/10 (previous run: 9.43/10, +0.00)
```
