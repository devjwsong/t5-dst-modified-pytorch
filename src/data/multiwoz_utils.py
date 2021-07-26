import re
import copy
import pickle


slot_descs = {
    "hotel-pricerange": "pricerange of the hotel",
    "hotel-type": "type of the hotel",
    "hotel-parking": "whether have parking in the hotel",
    "hotel-book stay": "number of stay for the hotel booking",
    "hotel-book day": "day for the hotel booking",
    "hotel-book people": "number of people for the hotel booking",
    "hotel-area": "area of the hotel",
    "hotel-stars": "number of stars of the hotel",
    "hotel-internet": "whether have internet in the hotel",
    "hotel-name": "name of the hotel",
    "train-destination": "location of destination of the train",
    "train-day": "day of the train",
    "train-departure": "location of departure of the train",
    "train-arriveBy": "time of arrive by of the train",
    "train-book people": "number of people for the train booking",
    "train-leaveAt": "time of leave at of the train",
    "attraction-type": "type of attraction",
    "attraction-area": "area of attraction",
    "attraction-name": "name of attraction",
    "restaurant-book people": "number of people for the restaurant booking",
    "restaurant-book day": "day for the restaurant booking",
    "restaurant-book time": "time for the restaurant booking",
    "restaurant-food": "food of the restaurant",
    "restaurant-pricerange": "pricerange of the restaurant",
    "restaurant-name": "name of the restaurant",
    "restaurant-area": "area of the restaurant",
    "taxi-leaveAt": "time of leave at of the taxi",
    "taxi-destination": "location of destination of the taxi",
    "taxi-departure": "location of departure of the taxi",
    "taxi-arriveBy": "time of arrive by of the taxi",
}


def get_domains(goal):
    domains = []
    for k, v in goal.items():
        if len(v) > 0 and k not in ['message', 'topic']:
            domains.append(k)
    return domains


def normalize_time(text):
    text = re.sub("(\d{1})(a\.?m\.?|p\.?m\.?)", r"\1 \2", text) # am/pm without space
    text = re.sub("(^| )(\d{1,2}) (a\.?m\.?|p\.?m\.?)", r"\1\2:00 \3", text) # am/pm short to long form
    text = re.sub("(^| )(at|from|by|until|after) ?(\d{1,2}) ?(\d{2})([^0-9]|$)", r"\1\2 \3:\4\5", text) # Missing separator
    text = re.sub("(^| )(\d{2})[;.,](\d{2})", r"\1\2:\3", text) # Wrong separator
    text = re.sub("(^| )(at|from|by|until|after) ?(\d{1,2})([;., ]|$)", r"\1\2 \3:00\4", text) # normalize simple full hour time
    text = re.sub("(^| )(\d{1}:\d{2})", r"\g<1>0\2", text) # Add missing leading 0
    # Map 12 hour times to 24 hour times
    text = re.sub("(\d{2})(:\d{2}) ?p\.?m\.?", lambda x: str(int(x.groups()[0]) + 12 if int(x.groups()[0]) < 12 else int(x.groups()[0])) + x.groups()[1], text)
    text = re.sub("(^| )24:(\d{2})", r"\g<1>00:\2", text) # Correct times that use 24 as hour
    return text


def normalize_text(text):
    text = normalize_time(text)
    text = re.sub("n't", " not", text)
    text = re.sub("(^| )zero(-| )star([s.,? ]|$)", r"\g<1>0 star\3", text)
    text = re.sub("(^| )one(-| )star([s.,? ]|$)", r"\g<1>1 star\3", text)
    text = re.sub("(^| )two(-| )star([s.,? ]|$)", r"\g<1>2 star\3", text)
    text = re.sub("(^| )three(-| )star([s.,? ]|$)", r"\g<1>3 star\3", text)
    text = re.sub("(^| )four(-| )star([s.,? ]|$)", r"\g<1>4 star\3", text)
    text = re.sub("(^| )five(-| )star([s.,? ]|$)", r"\g<1>5 star\3", text)
    text = re.sub("archaelogy", "archaeology", text) # Systematic typo
    text = re.sub("guesthouse", "guest house", text) # Normalization
    text = re.sub("(^| )b ?& ?b([.,? ]|$)", r"\1bed and breakfast\2", text) # Normalization
    text = re.sub("bed & breakfast", "bed and breakfast", text) # Normalization
    return text


# This should only contain label normalizations. All other mappings should
# be defined in LABEL_MAPS.
def normalize_label(slot, value_label):
    # Normalization of empty slots
    if value_label == '' or value_label == "not mentioned":
        return "none"

    # Normalization of time slots
    if "leaveAt" in slot or "arriveBy" in slot or slot == 'restaurant-book_time':
        return normalize_time(value_label)

    # Normalization
    if "type" in slot or "name" in slot or "destination" in slot or "departure" in slot:
        value_label = re.sub("guesthouse", "guest house", value_label)

    # Map to boolean slots
    if slot == 'hotel-parking' or slot == 'hotel-internet':
        if value_label == 'yes' or value_label == 'free':
            return "true"
        if value_label == "no":
            return "false"
    if slot == 'hotel-type':
        if value_label == "hotel":
            return "true"
        if value_label == "guest house":
            return "false"

    return value_label


def save_files(to_dir, prefix, utters, states):
    with open(f"{to_dir}/{prefix}_utters.pickle", 'wb') as f:
        pickle.dump(utters, f)
        
    with open(f"{to_dir}/{prefix}_states.pickle", 'wb') as f:
        pickle.dump(states, f)
        

def count_utters(dialogues):
    num = 0
    for dialogue in dialogues:
        num += len(dialogue)
        
    return num
