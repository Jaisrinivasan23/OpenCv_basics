from snake.game import Game, GameConf, GameMode

greedy = "Greedysolver"
hamilton = "Hamiltonsolver"
normal = GameMode.NORMAL
conf = GameConf(10, 10, normal, greedy)
conf.solve()
conf.model = normal

print("Game Model: ", conf.model)
print("Game Mode: ", conf.mode)
Game(conf).run()
