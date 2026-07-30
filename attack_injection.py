import random


class AttackInjection:

    def __init__(self, nodes):
        self.nodes = nodes

    def assign_attack_nodes(self):

        total = len(self.nodes)

        indices = list(range(total))
        random.shuffle(indices)

        blackhole_count = int(total * 0.10)
        grayhole_count = int(total * 0.10)
        dos_count = int(total * 0.10)

        blackhole = indices[:blackhole_count]
        grayhole = indices[
            blackhole_count:
            blackhole_count + grayhole_count
        ]
        dos = indices[
            blackhole_count + grayhole_count:
            blackhole_count + grayhole_count + dos_count
        ]

        for node in self.nodes:
            node.attack_type = "Normal"

        for i in blackhole:
            self.nodes[i].attack_type = "Blackhole"

        for i in grayhole:
            self.nodes[i].attack_type = "Grayhole"

        for i in dos:
            self.nodes[i].attack_type = "DoS"

        print("\nAttack Injection Completed")

        print("Blackhole :", blackhole_count)
        print("Grayhole  :", grayhole_count)
        print("DoS       :", dos_count)
        print("Normal    :", total-blackhole_count-grayhole_count-dos_count)