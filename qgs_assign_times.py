import datetime
def main():
    layers = QgsProject.instance().mapLayers()
    
    dates = []
    
    for key in layers:
        layer = layers[key]
        
        name = layer.name()
        if not any(s in name for s in ["ASCENDING", "DESCENDING"]):
            continue
        date = datetime.datetime.strptime(name.split("_")[-1], "%Y-%m-%dT%H-%M-%S")
        
        dates.append([key, date])
        
        continue
        
    dates.sort(key=lambda d: d[1])
    
    print(dates)
    
    for i, (key, date) in enumerate(dates):
        
        if i == (len(dates) - 1):
            other_date = date
        else:
            other_date = dates[i + 1][1]
        layer = layers[key]
        temporal = layer.temporalProperties()
        temporal.setMode(QgsRasterLayerTemporalProperties.ModeFixedTemporalRange)
        temporal.setFixedTemporalRange(QgsDateTimeRange(date, date))
        temporal.setIsActive(True)
    
    print("Done assigning times")
    
main()