import numpy as np

#{(stationid,1);(city,"Bucharest");(temp,15);(rain,0.5);(wind,12);(direction,"NE");(date,2.02.2023)}
class Publication:
    def __init__(self, station_id=None, city=None, temp=None, rain=None, wind=None, direction=None, date=None):
        self.station_id = station_id
        self.city = city
        self.temp = temp
        self.rain = wind
        self.direction = direction
        self.date = date
    
    def __str__(self):
        return ''.join((
            f"(stationid, {self.station_id}); ",
            f"(city, '{self.city}'); ",
            f"(temp, {self.temp}); ",
            f"(rain, {self.rain}); ",
            f"(wind, {self.wind}); ",
            f"(direction, '{self.direction}'); ",
            f"(date, {self.date})"))

#stationid, city, direction, date -> prestabilite
#temp, wind, rain -> random intre limite
class PublicationGenerator:
    def __init__(self, station_pool, city_pool, direction_pool, date_pool, temp_limits, wind_limits, rain_limits):
        self.station_pool = station_pool
        self.city_pool = city_pool
        self.direction_pool = direction_pool
        self.date_pool = date_pool
        self.temp_limits = temp_limits
        self.wind_limits = wind_limits
        self.rain_limits = rain_limits
        
    def generate_publication(self):
        new_publication = Publication()
        new_publication.station_id = self.station_pool[np.random.randint(0, len(self.station_pool))]
        new_publication.city = self.city_pool[np.random.randint(0, len(self.city_pool))]
        new_publication.direction = self.direction_pool[np.random.randint(0, len(self.direction_pool))]
        new_publication.date = self.date_pool[np.random.randint(0, len(self.date_pool))]
        new_publication.temp = np.random.randint(self.temp_limits[0], self.temp_limits[1])
        new_publication.wind = np.random.randint(self.wind_limits[0], self.wind_limits[1])
        new_publication.rain = round(np.random.uniform(self.rain_limits[0], self.rain_limits[1]), 2)
        return new_publication
            
    def generate_publications(self, number):
        return [self.generate_publication() for _ in range(number)]
    

if __name__ == "__main__":
    stations = [1, 4, 7, 15, 23, 27, 69, 100]
    cities = ["Bucharest", "Harlau", "Braila", "Galati", "Darabani", "Dubai"]
    directions = ["N", "NE", "S", "NV", "3SE", "X"]
    dates = [f"{i}.04.2023" for i in range(10, 17)]
    temp_limits = (-20, 40)
    wind_limits = (0, 100)
    rain_limits = (0, 1)
    
    generator = PublicationGenerator(stations, cities, directions, dates, temp_limits, wind_limits, rain_limits)
    publications = generator.generate_publications(6)
    print(len(publications))
    print(*[f"{i}\n" for i in publications], sep='\n')