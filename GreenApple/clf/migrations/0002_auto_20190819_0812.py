# Generated by Django 2.2 on 2019-08-19 08:12

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('clf', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='fruitimage',
            name='uploaded_at',
            field=models.DateTimeField(auto_now_add=True, default=django.utils.timezone.now),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='fruitimage',
            name='fruit_image',
            field=models.ImageField(upload_to='clf/Images/'),
        ),
    ]