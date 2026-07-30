class AttackBehavior:

    def __init__(self):
        pass

    def classify_node(
        self,
        trust,
        packet_delivery_ratio,
        energy,
        transmission_rate
    ):

        # -------------------------
        # Blackhole
        # -------------------------
        if (
            trust < 0.30
            and packet_delivery_ratio < 0.20
        ):
            return "Blackhole"

        # -------------------------
        # Grayhole
        # -------------------------
        elif (
            trust < 0.70
            and packet_delivery_ratio < 0.70
        ):
            return "Grayhole"

        # -------------------------
        # DoS
        # -------------------------
        elif (
            transmission_rate > 80
            and energy < 0.30      # Change to 30 if your energy scale is 0-100
        ):
            return "DoS"

        # -------------------------
        # Normal
        # -------------------------
        else:
            return "Normal"