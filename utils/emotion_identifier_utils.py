def identify_emotion(emotion):
    if not isinstance(emotion, str):
        return None  

    emotion = emotion.strip().lower()  

    match emotion:
        case "ang" | "05" | "a" | "angry" | "anger":
            return "angry"
        case "dis" | "07" | "d" | "disgust":
            return "disgust"
        case "fea" | "06" | "f" | "fear":
            return "fear"
        case "hap" | "03" | "h" | "happy":
            return "happy"
        case "neu" | "01" | "n" | "neutral":
            return "neutral"
        case "sad" | "04" | "sa" | "sad":
            return "sad"
        case "surprised" | "su" | "08":
            return "surprised"
        case _:
            return "unknown"