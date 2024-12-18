from Model import train

if __name__ == "__main__":
    args = train.parse_args()
    trainer = train.Trainer("sd-legacy/stable-diffusion-v1-5", "data/image_data.json", args=args)
    trainer.train(args=args)