# Generated by Django 4.2.1 on 2025-05-13 17:45

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("account_v2", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="user",
            name="auth_provider",
            field=models.CharField(default="", max_length=64),
        ),
    ]
