CLIMATE_CHUNK_INSTRUCTION = \
"""
You are an expert in climate forecasting.
Extract and summarize all relevant information to create a new weather forecast including valid dates.
Ensure all relevant information and context is captured accurately. Include forecast periods and relevant dates."
Avoid extraneous phrases such as ""here is the new summarized weather forecast.""
"""

CLIMATE_REFINER_INSTRUCTION = \
"""
You are tasked with assesing if the summarized doctor's medical notes captures the original notes.
Based on the original text, provide a new summarized medical notes of all information, observations, and predictions related to respiratory rate, heart rate, SaO2, and FiO2.
Output the new meidcal notes without extraneous phrases.
"""

CLIMATE_REFINER_TEMPLATE = \
"""
`````
Original_weather_forecast:
`````
{input}
`````
summarized_weather_forecast:
`````
{summary}
`````
new_summarized_weather_forecast:
"""

MEDICAL_CHUNK_INSTRUCTION = \
"""
You are an expert in doctor's medical notes and can use knowledge from MIMIC III dataset.
Extract and summarize all relevant information to create a new medical note.
Ensure all relevant information and context is captured accurately.
Avoid extraneous phrases such as here is the new summarized medical note and leave blank if there is no relevant information.
"""

MEDICAL_REFINER_INSTRUCTION = \
"""
You are tasked with assesing if the summarized doctor's medical notes captures the original notes.
Based on the original text, provide a new summarized medical notes of all information, observations, and predictions related to respiratory rate, heart rate, SaO2, and FiO2.
Output the new meidcal notes without extraneous phrases.
"""

MEDICAL_REFINER_TEMPLATE = \
"""
`````
Original_medical_note:
`````
{input}
`````
summarized_medical_note:
`````
{summary}
`````
new_summarized_medical_note:
"""