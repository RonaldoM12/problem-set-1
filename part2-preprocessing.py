'''
PART 2: Pre-processing
- Take the time to understand the data before proceeding
- Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
- Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
- Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`. 
- - So if a person was arrested on 2016-09-11, you would check to see if there was a felony arrest for that person between 2016-09-12 and 2017-09-11.
- - Use a print statment to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?
- Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise. 
- - Use a print statment to print this question and its answer: What share of current charges are felonies?
- Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year` which is the total number arrests in the one year prior to the current charge. 
- - So if someone was arrested on 2016-09-11, then you would check to see if there was a felony arrest for that person between 2015-09-11 and 2016-09-10.
- - Use a print statment to print this question and its answer: What is the average number of felony arrests in the last year?
- Print the mean of 'num_fel_arrests_last_year' -> pred_universe['num_fel_arrests_last_year'].mean()
- Print pred_universe.head()
- Return `df_arrests` for use in main.py for PART 3; if you can't figure this out, save as a .csv in `data/` and read into PART 3 in main.py
'''

# import the necessary packages
import pandas as pd
from datetime import timedelta

# Your code here
def preprocess_data():
    pred_universe_raw = pd.read_csv("data/pred_universe_raw.csv", parse_dates=['arrest_date_univ'])
    arrest_events_raw = pd.read_csv("data/arrest_events_raw.csv", parse_dates=['arrest_date_event'])

    df_arrests = pd.merge(pred_universe_raw, arrest_events_raw, how='outer', on='person_id')

    for x, row in df_arrests.iterrows():
        y = row['person_id']
        date = row['arrest_date_univ']


        if pd.isnull(date):
            continue

        one_day_after = date + timedelta(days=1)
        one_year_after = date + timedelta(days=365)

        arrests = arrest_events_raw[
            (arrest_events_raw['person_id'] == y) &
            (arrest_events_raw['arrest_date_event'] >= one_day_after) &
            (arrest_events_raw['arrest_date_event'] <= one_year_after) &
            (arrest_events_raw['top_charge_type'] == "F")
        ]

        if not arrests.empty:
            df_arrests.at[x, 'y'] = 1

    rearrested = df_arrests['y'].mean()
    print(f"What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?")


    df_arrests['current_change_felony'] = (df_arrests['top_charge_type_x'] == 'F')


    current_Felony = df_arrests['current_charge_felony'].mean()

    print(f"What share of current charges are felonies? {current_Felony:.3f}")

    def count_pastFelony(row):
        y = row['person_id']
        arrested_date = row['arrest_date_univ']

        if pd.isnull(arrested_date):
            return 0
        
        one_year_before = arrested_date - timedelta(days=365)
        one_day_before = arrested_date = timedelta(days=1)

        past_felonies = arrest_events_raw[
            (arrest_events_raw['person_id'] == y) &
                (arrest_events_raw['arrest_date_event'] >= one_year_before) &
                (arrest_events_raw['arrest_date_event'] <= one_day_after) &
                (arrest_events_raw['top_charge_type'] == "F")
                ]
        
        return past_felonies.shape[0]

    df_arrests['num_fel_arrests_last_year'] = df_arrests.apply(count_pastFelony, axis=1)


    avg_felony = df_arrests['num_fel_arrests_last_year'].mean()
    print(f"What is the average number of felony arrests in the last year? {avg_felony:.3f}")

    print("Mean of 'num_fel_arrests_last_year': ", df_arrests['num_fel_arrests_last_year'].mean())

    print(df_arrests.head())

    df_arrests.to_csv("data/df_arrests.csv", index=False)

