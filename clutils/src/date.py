import datetime

class GMT1(datetime.tzinfo):
    def utcoffset(self, dt):
        return datetime.timedelta(hours=1)
    def dst(self, dt):
        return datetime.timedelta(0)
    def tzname(self,dt):
        return "Europe/Paris"
