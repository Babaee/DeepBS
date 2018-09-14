# bash command for renaming all files within a folder to following format bin%06d.png
# USE: background images, created with bgslibrary -> 1.png - 100.png -> bin000001.ong - bin 0000100.png


for file in [0-9]*.png; do
  # strip the prefix ("foo") off the file name
  #postfile=${file#foo}
  # strip the postfix (".png") off the file name
  number=${file%.png}
  # subtract 1 from the resulting number
  # copy to a new name in a new folder
  # mv ${file} $(printf bin%06d.png $number)
  mv ${file} $(printf bin%06d.png $number)
done
