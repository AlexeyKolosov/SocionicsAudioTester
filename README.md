# SocionicsAudioTester
python tool for determining the strong socionic type aspects based on Russian speech

Quick algorithm:
- Russian speech recording
- Speech to text recognizing
- Text to words and n-grams splitting
- Every word and n-gram classification through the FastText-model, trained with 8 dictionaries, prepared in [Кочубеева Л.А., Миронов В.В., Стоялова М.Л. "Соционика. Семантика информационных аспектов"]
- Retrieving the result
