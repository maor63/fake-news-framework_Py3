import botometer

class BotometerObj():
    def __init__(self):
        mashape_key = "IvqHr43buUmshDE8PIlfX7rurSg1p1RKwRnjsnalA69vCu7vG6"
        twitter_app_auth = {
            'consumer_key': 'YewkYt0uOr8XLHEKeoCm5IljS',
            'consumer_secret': 'qWcYeoJ1ebUScGhW9V8yseJFbaGa7wt3eaI0nceM6XHbtwqvmd',
            'access_token': '714699220352155648-3QXFPHQC7yEjK3T9BQuEIJ7Rny8I3q3',
            'access_token_secret': 'UroyUvZVfl3guYdFKnkqdkNytkUhIKwLcw46kKHTfFQwY',
        }
        self._bom = botometer.Botometer(wait_on_ratelimit=True,
                                            mashape_key=mashape_key,
                                            **twitter_app_auth)

    def get_botometer(self):
        return self._bom