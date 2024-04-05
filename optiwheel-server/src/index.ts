import ansiColors from "ansi-colors";
import express, { NextFunction, Request, Response } from "express";
import { Server } from "http";
import { z } from "zod";

const app = express();

const PORT: string | 8080 = 8080;

interface Direction {
  x: number;
  y: number;
}

let dir: Direction = {
  x: 0,
  y: 0,
};

app.use(
  express.urlencoded({
    extended: false,
  })
);
app.use(express.json({}));

app.get("/", (req: Request, res: Response, next: NextFunction): void => {
  console.table({ dir });

  res.send({
    data: { dir },
    success: true,
  });
});

app.post("/", (req: Request, res: Response, next: NextFunction): void => {
  const data: any = req.body;

  const parsedData: z.SafeParseReturnType<Direction, Direction> =
    DirectionSchema.safeParse(data);

  if (!parsedData.success) {
    res.send({
      success: false,
      error: parsedData.error,
    });
  } else {
    dir = parsedData.data;
    console.table({ dir });
    res.send({
      data: { dir },
      success: true,
    });
  }
});

app.listen(PORT, function (): void {
  console.log("running");
});

const DirectionSchema: z.ZodObject<
  {
    x: z.ZodNumber;
    y: z.ZodNumber;
  },
  "strip",
  z.ZodTypeAny,
  Direction,
  Direction
> = z.object({
  x: z.number().min(-1).max(1),
  y: z.number().min(0).max(1),
});
