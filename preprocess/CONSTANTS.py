# Columns description
'''
id                 : primary key of the table. 
match_date         : when the match takes place
kick_off_time      : time each match starts 
section            : the round of the season as we know it. In the 
                     dataset, denotes it as section, thus we comply
                     to the naming system 
round              : how many days passed 
                     from the very first day on that section
home_team          : name of the hometeam
away_team          : name of the awayteam
stadium            : where the match takes place
weather            : how was the weather when the match held
temperature        : Farenheit, the temperate when the match held
humidity           : humidity when the match held
broadcaster        : name of the broadcast chaneel 
capacity           : how many people can the stadium can accomodate
description        : denote the name of the holiday if the match held
                     on holiday
is_holiday         : whether the match held in holiday (1 : Yes, 0  : False)
is_dayoff          : whether the match held in dayoff (1 : Yes, 0  : False)
After_is_dayoff    : How many days past from the last dayoff
Before_is_dayoff   : How many days left before the next dayoff
After_is_holiday   : How many days past from the last holiday
Before_is_holiday  : How many days left before the next hoilday 
'''