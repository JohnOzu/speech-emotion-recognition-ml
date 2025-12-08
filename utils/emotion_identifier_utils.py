def identify_emotion(emotion):
    if not isinstance(emotion, str):
        return None  

    emotion = emotion.strip().lower()  

    match emotion:
        case "ang" | "05" | "a" | "angry" | "anger" | "rab" | "W":
            return "angry"
        case "dis" | "07" | "d" | "disgust" | "E":
            return "disgust"
        case "fea" | "06" | "f" | "fear" | "pau" | "A":
            return "fear"
        case "hap" | "03" | "h" | "happy" | "gio" | "F":
            return "happy"
        case "neu" | "01" | "n" | "neutral" | "neut" | "N":
            return "neutral"
        case "sad" | "04" | "sa" | "sad" | "tri" | "T":
            return "sad"
        case "surprised" | "su" | "08":
            return "surprised"
        case _:
            return "unknown"